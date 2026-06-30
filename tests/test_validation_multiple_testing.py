def test_trial_ledger_accumulates():
    from factorzen.validation.multiple_testing import TrialLedger
    led = TrialLedger()
    assert led.n_trials == 0
    led.record()
    led.record(5)
    assert led.n_trials == 6


def test_trial_ledger_default_zero():
    from factorzen.validation.multiple_testing import TrialLedger
    assert TrialLedger().n_trials == 0
