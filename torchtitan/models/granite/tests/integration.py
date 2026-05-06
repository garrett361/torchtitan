from tests.integration_tests import OverrideDefinitions


def build_granite_tests_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            [
                [
                    "--module granite --config granite_debugmodel_float8_rowwise",
                ],
            ],
            "Granite Float8 rowwise",
            "granite_float8_rowwise",
        ),
        OverrideDefinitions(
            [
                [
                    "--module granite --config granite_debugmodel_float8_rowwise",
                    "--compile.enable",
                    "--parallelism.context_parallel_degree 2",
                ],
            ],
            "Granite Float8 rowwise + CP + compile",
            "granite_float8_rowwise+cp+compile",
        ),
    ]
