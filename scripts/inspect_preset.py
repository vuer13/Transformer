from transformer.model import Model
from transformer.presets import get_80m_config, get_local_debug_config
from transformer.training import count_parameters


def print_model_size(name: str, model: Model) -> None:
    parameter_count = count_parameters(model)

    print(f"{name}")
    print(f"parameters: {parameter_count:,}")
    print(f"parameters in millions: {parameter_count / 1_000_000:.2f}M")
    print()


def main() -> None:
    local_config = get_local_debug_config(vocab_size=50257)
    local_model = Model(local_config)

    large_config = get_80m_config(vocab_size=50257)
    large_model = Model(large_config)

    print_model_size("local debug config", local_model)
    print_model_size("80M experiment config", large_model)


if __name__ == "__main__":
    main()
