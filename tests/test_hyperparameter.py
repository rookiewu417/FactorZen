"""Optuna 超参搜索测试。"""

from __future__ import annotations

from daily.evaluation.hyperparameter import ParamSpec, TuningSpace, run_optuna_search

# ── TestTuningSpace ──────────────────────────────────────────────────────────


class TestTuningSpace:
    def test_suggest_int(self):
        """整数参数 suggest 应返回指定范围内的 int。"""
        import optuna

        space = TuningSpace([ParamSpec("n_groups", "int", low=5, high=20)])
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        assert "n_groups" in params
        assert isinstance(params["n_groups"], int)
        assert 5 <= params["n_groups"] <= 20

    def test_suggest_float(self):
        """浮点参数 suggest 应返回指定范围内的 float。"""
        import optuna

        space = TuningSpace([ParamSpec("lr", "float", low=0.001, high=0.1)])
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        assert "lr" in params
        assert isinstance(params["lr"], float)
        assert 0.001 <= params["lr"] <= 0.1

    def test_suggest_categorical(self):
        """分类参数 suggest 应返回 choices 之一。"""
        import optuna

        choices = ["a", "b", "c"]
        space = TuningSpace([ParamSpec("mode", "categorical", choices=choices)])
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        assert "mode" in params
        assert params["mode"] in choices

    def test_suggest_all_types(self):
        """混合类型的 TuningSpace 应返回所有参数键。"""
        import optuna

        space = TuningSpace(
            [
                ParamSpec("n", "int", low=10, high=50),
                ParamSpec("alpha", "float", low=0.01, high=1.0),
                ParamSpec("method", "categorical", choices=["a", "b"]),
            ]
        )
        study = optuna.create_study()
        trial = study.ask()
        params = space.suggest(trial)
        assert set(params.keys()) == {"n", "alpha", "method"}


# ── TestRunOptunaSearch ──────────────────────────────────────────────────────


class TestRunOptunaSearch:
    def test_finds_optimum_on_convex(self):
        """凸目标函数 -(x-3)^2 的最大值应在 x=3 附近（允许误差 1.0）。"""
        space = TuningSpace([ParamSpec("x", "float", low=0.0, high=6.0)])

        def objective(params: dict) -> float:
            return -((params["x"] - 3.0) ** 2)

        best_params, _study = run_optuna_search(
            objective_fn=objective,
            space=space,
            n_trials=20,
            direction="maximize",
            study_name="test_convex",
        )
        assert "x" in best_params
        assert abs(best_params["x"] - 3.0) < 1.0, (
            f"最优 x={best_params['x']:.3f} 与真实最优 3.0 相差超过 1.0"
        )

    def test_returns_best_params_dict(self):
        """run_optuna_search 应返回包含期望键的字典。"""
        space = TuningSpace([ParamSpec("n", "int", low=1, high=10)])

        def objective(params: dict) -> float:
            return float(params["n"])

        best_params, _study = run_optuna_search(
            objective_fn=objective,
            space=space,
            n_trials=5,
            direction="maximize",
            study_name="test_returns_dict",
        )
        assert isinstance(best_params, dict)
        assert "n" in best_params

    def test_direction_minimize(self):
        """direction="minimize" 时应最小化目标函数。"""
        space = TuningSpace([ParamSpec("x", "float", low=0.0, high=10.0)])

        def objective(params: dict) -> float:
            return (params["x"] - 5.0) ** 2

        best_params, study = run_optuna_search(
            objective_fn=objective,
            space=space,
            n_trials=20,
            direction="minimize",
            study_name="test_minimize",
        )
        assert abs(best_params["x"] - 5.0) < 2.0, (
            f"minimize 时最优 x={best_params['x']:.3f} 应接近 5.0"
        )
        # 最优值应为最小（接近 0）
        assert study.best_value < 5.0
