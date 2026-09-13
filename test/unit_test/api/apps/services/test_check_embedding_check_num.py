#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Regression tests for check_num validation in check_embedding() (#19567)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.p2


def _install_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, child_name, module, raising=False)
    return module


class _FakeSearchResult:
    pass


class _ValidVectorStore:
    """Doc store with one comparable embedded chunk."""

    def db_type(self):
        return "elasticsearch"

    def search(self, *_args, **_kwargs):
        return _FakeSearchResult()

    def get_total(self, _res):
        return 1

    def get_doc_ids(self, _res):
        return ["chunk-1"]

    def get(self, _chunk_id, _index_name, _kb_ids):
        return {
            "id": "chunk-1",
            "kb_id": "kb1",
            "doc_id": "doc1",
            "docnm_kwd": "doc.txt",
            "content_with_weight": "sample text",
            "q_4_vec": [1.0, 0.0, 0.0, 0.0],
        }


class _CompatibleEmbeddingBundle:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, *_args, **_kwargs):
        import numpy as np

        return [np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])], None


def _load_dataset_module(monkeypatch, *, doc_store=None):
    repo_root = Path(__file__).resolve().parents[5]
    kb = SimpleNamespace(id="kb1", tenant_id="tenant1", embd_id="old-embd")

    _install_module(
        monkeypatch,
        "api.db.joint_services.tenant_model_service",
        get_model_config_from_provider_instance=lambda *_args, **_kwargs: {"llm_name": "new-embd"},
        get_composite_model_name_by_ids=lambda _ids: {},
        resolve_model_config=lambda *_args, **_kwargs: {"llm_name": "new-embd"},
        resolve_model_id=lambda *_args, **_kwargs: "new-embd",
    )
    _install_module(
        monkeypatch,
        "api.db.db_models",
        Connector2Kb=SimpleNamespace(kb_id="kb_id"),
        Document=SimpleNamespace(kb_id="kb_id"),
        File=SimpleNamespace(),
        SyncLogs=SimpleNamespace(kb_id="kb_id", status=SimpleNamespace(in_=lambda _values: None)),
    )
    _install_module(
        monkeypatch,
        "api.db.services.document_service",
        DocumentService=SimpleNamespace(),
        queue_raptor_o_graphrag_tasks=MagicMock(),
    )
    _install_module(monkeypatch, "api.db.services.file2document_service", File2DocumentService=SimpleNamespace())
    _install_module(monkeypatch, "api.db.services.file_service", FileService=SimpleNamespace())
    _install_module(
        monkeypatch,
        "api.db.services.knowledgebase_service",
        KnowledgebaseService=SimpleNamespace(
            accessible=staticmethod(lambda *_args: True),
            get_by_id=staticmethod(lambda _dataset_id: (True, kb)),
        ),
        validate_dataset_embedding_models=lambda *_args, **_kwargs: None,
    )
    _install_module(
        monkeypatch,
        "api.db.services.connector_service",
        Connector2KbService=SimpleNamespace(),
        SyncLogsService=SimpleNamespace(),
    )
    _install_module(
        monkeypatch,
        "api.db.services.task_service",
        GRAPH_RAPTOR_FAKE_DOC_ID="fake-doc",
        TaskService=SimpleNamespace(),
    )
    _install_module(
        monkeypatch,
        "api.db.services.user_service",
        TenantService=SimpleNamespace(),
        UserService=SimpleNamespace(),
        UserTenantService=SimpleNamespace(),
    )
    _install_module(
        monkeypatch,
        "api.db.services.tenant_model_service",
        TenantModelService=SimpleNamespace(get_by_id=staticmethod(lambda *_args, **_kwargs: (True, None))),
    )
    _install_module(monkeypatch, "common.settings", docStoreConn=doc_store or _ValidVectorStore())
    _install_module(
        monkeypatch,
        "common.constants",
        FileSource=SimpleNamespace(KNOWLEDGEBASE="knowledgebase"),
        LLMType=SimpleNamespace(EMBEDDING="embedding"),
        PAGERANK_FLD="pagerank_fea",
        RetCode=SimpleNamespace(NOT_EFFECTIVE=590),
        StatusEnum=SimpleNamespace(VALID=SimpleNamespace(value="1")),
        TaskStatus=SimpleNamespace(SCHEDULE="schedule", RUNNING="running", CANCEL="cancel"),
    )
    _install_module(
        monkeypatch,
        "api.utils.api_utils",
        deep_merge=lambda base, update: {**(base or {}), **(update or {})},
        get_parser_config=lambda *_args, **_kwargs: {},
        remap_dictionary_keys=lambda value: value,
        verify_embedding_availability=lambda *_args, **_kwargs: (True, ""),
    )
    _install_module(
        monkeypatch,
        "api.db.services.llm_service",
        LLMBundle=_CompatibleEmbeddingBundle,
        resolve_llm_setting=lambda *_args, **_kwargs: {},
    )
    rag = _install_module(monkeypatch, "rag")
    rag.__path__ = []
    advanced_rag = _install_module(monkeypatch, "rag.advanced_rag")
    advanced_rag.__path__ = []
    knowledge_compile = _install_module(monkeypatch, "rag.advanced_rag.knowlege_compile")
    knowledge_compile.__path__ = []
    _install_module(monkeypatch, "rag.advanced_rag.knowlege_compile.wiki", WIKI_PAGE_COMPILE_KWD="artifact_page")
    _install_module(monkeypatch, "rag.nlp")
    _install_module(monkeypatch, "rag.nlp.search", index_name=lambda tenant_id: f"ragflow_{tenant_id}")

    spec = importlib.util.spec_from_file_location(
        "api.apps.services.dataset_api_service",
        repo_root / "api" / "apps" / "services" / "dataset_api_service.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "api.apps.services.dataset_api_service", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "check_num",
    ["invalid", "1.5", None, [], {}],
)
def test_check_embedding_rejects_non_integer_check_num(monkeypatch, check_num):
    module = _load_dataset_module(monkeypatch)

    ok, result = module.check_embedding("kb1", "tenant1", {"embd_id": "new-embd", "check_num": check_num})

    assert ok is False
    assert result == "`check_num` should be an integer"


@pytest.mark.parametrize("check_num", [0, -1, -10])
def test_check_embedding_rejects_non_positive_check_num(monkeypatch, check_num):
    module = _load_dataset_module(monkeypatch)

    ok, result = module.check_embedding("kb1", "tenant1", {"embd_id": "new-embd", "check_num": check_num})

    assert ok is False
    assert result == "`check_num` must be greater than 0."


def test_check_embedding_defaults_check_num_when_missing(monkeypatch):
    module = _load_dataset_module(monkeypatch)

    ok, result = module.check_embedding("kb1", "tenant1", {"embd_id": "new-embd"})

    assert ok is True
    assert result["summary"]["sampled"] == 1
    assert result["summary"]["valid"] == 1


def test_check_embedding_accepts_positive_check_num(monkeypatch):
    module = _load_dataset_module(monkeypatch)

    ok, result = module.check_embedding("kb1", "tenant1", {"embd_id": "new-embd", "check_num": 3})

    assert ok is True
    assert result["summary"]["sampled"] == 1
    assert result["summary"]["avg_cos_sim"] == 1.0
