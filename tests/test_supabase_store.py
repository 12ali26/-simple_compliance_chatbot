from src.supabase_store import SupabaseStore


class FakeResponse:
    data = [{"id": "row-1"}]


class FakeQuery:
    def __init__(self, table_name, calls):
        self.table_name = table_name
        self.calls = calls

    def select(self, columns):
        self.calls.append((self.table_name, "select", columns))
        return self

    def update(self, payload):
        self.calls.append((self.table_name, "update", payload))
        return self

    def delete(self):
        self.calls.append((self.table_name, "delete", None))
        return self

    def eq(self, column, value):
        self.calls.append((self.table_name, "eq", column, value))
        return self

    def order(self, column, desc=False):
        self.calls.append((self.table_name, "order", column, desc))
        return self

    def limit(self, count):
        self.calls.append((self.table_name, "limit", count))
        return self

    def execute(self):
        self.calls.append((self.table_name, "execute", None))
        return FakeResponse()


class FakeClient:
    def __init__(self):
        self.calls = []

    def table(self, table_name):
        self.calls.append((table_name, "table", None))
        return FakeQuery(table_name, self.calls)


def make_store():
    store = SupabaseStore.__new__(SupabaseStore)
    store.settings = object()
    store.client = FakeClient()
    return store


def test_document_crud_uses_expected_tables_and_filters():
    store = make_store()

    assert store.list_documents() == [{"id": "row-1"}]
    store.update_document("doc-1", {"title": "New", "ignored": "Nope"})
    store.delete_document("doc-1")

    assert ("documents", "select", "*") in store.client.calls
    assert ("documents", "update", {"title": "New"}) in store.client.calls
    assert ("documents", "delete", None) in store.client.calls
    assert store.client.calls.count(("documents", "eq", "id", "doc-1")) == 2


def test_log_crud_uses_expected_tables_and_filters():
    store = make_store()

    assert store.list_chat_logs() == [{"id": "row-1"}]
    store.delete_chat_log("chat-1")
    assert store.list_unanswered_questions() == [{"id": "row-1"}]
    store.delete_unanswered_question("question-1")

    assert ("chat_logs", "select", "*") in store.client.calls
    assert ("chat_logs", "delete", None) in store.client.calls
    assert ("chat_logs", "eq", "id", "chat-1") in store.client.calls
    assert ("unanswered_questions", "select", "*") in store.client.calls
    assert ("unanswered_questions", "delete", None) in store.client.calls
    assert ("unanswered_questions", "eq", "id", "question-1") in store.client.calls
