import pytest

from cti.protocol import ServingEndpoints


def test_every_links_receiver_binds_it() -> None:
    endpoints = ServingEndpoints.create(num_tokenizers=1)

    for target in ("engine", "frontend", "detokenizer", "tokenizer"):
        address, _ = endpoints.link(target, receiving=True)
        assert endpoints.link(target, receiving=True) == (address, True)
        assert endpoints.link(target, receiving=False) == (address, False)
    assert len({endpoints.engine, endpoints.frontend, endpoints.detokenizer}) == 3


def test_each_tokenizer_gets_its_own_address() -> None:
    endpoints = ServingEndpoints.create(num_tokenizers=2)

    assert endpoints.num_tokenizers == 2
    first, second = (endpoints.link("tokenizer", receiving=True, index=i)[0] for i in range(2))
    assert first != second
    assert endpoints.link("tokenizer", receiving=False, index=1) == (second, False)
    assert len({endpoints.engine, endpoints.frontend, endpoints.detokenizer, first, second}) == 5


def test_an_offline_engine_has_no_tokenizer_link() -> None:
    with pytest.raises(ValueError, match="no tokenizer 0 of 0"):
        ServingEndpoints.create().link("tokenizer", receiving=True)


def test_each_server_gets_fresh_addresses() -> None:
    first, second = ServingEndpoints.create(), ServingEndpoints.create()

    assert {first.engine, first.frontend}.isdisjoint({second.engine, second.frontend})
    assert all(address.startswith("ipc://") for address in (first.engine, first.frontend))
