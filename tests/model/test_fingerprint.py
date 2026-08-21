"""A Recording's fingerprint is what makes staleness detectable at all: any
input that would change the answer must change the hash."""


def test_identical_requests_fingerprint_identically(make_request):
    assert make_request().fingerprint() == make_request().fingerprint()


def test_recording_id_does_not_affect_the_fingerprint(make_request):
    assert make_request(recording_id="a").fingerprint() == make_request(recording_id="b").fingerprint()


def test_a_prompt_version_change_changes_the_fingerprint(make_request):
    assert (
        make_request(prompt_version="v1").fingerprint()
        != make_request(prompt_version="v2").fingerprint()
    )


def test_a_model_change_changes_the_fingerprint(make_request):
    assert (
        make_request(model="llama-3.1-8b").fingerprint()
        != make_request(model="llama-3.1-70b").fingerprint()
    )


def test_a_tool_schema_change_changes_the_fingerprint(make_request):
    with_tool = make_request(tools=[{"name": "escalate", "parameters": {}}])
    without_tool = make_request(tools=[])

    assert with_tool.fingerprint() != without_tool.fingerprint()


def test_a_message_content_change_changes_the_fingerprint(make_request):
    a = make_request(messages=[{"role": "user", "content": "hi"}])
    b = make_request(messages=[{"role": "user", "content": "hello"}])

    assert a.fingerprint() != b.fingerprint()
