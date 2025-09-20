from tests.chat_models import test_transformers_prompt_improve, test_transformers_chat, test_transformers_prompt_improve_similar
from tests.mixed_tests import test_curate, test_danbooru_curate

if __name__ == "__main__":
    # Test transformers chat
    test_transformers_chat()
    test_transformers_prompt_improve()
    test_transformers_prompt_improve_similar()

    # Test curate
    test_curate()
    test_danbooru_curate()

