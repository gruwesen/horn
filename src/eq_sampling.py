def run_seeded_then_random_eq(H, eval_formula, set2formula_fn,
                              generate_positive_sample, generate_sample,
                              n_positive_per_eq, n_eq_samples, bad_nc=None):
    h = set2formula_fn(H)
    sample_nr = 0
    bad_nc = bad_nc if bad_nc is not None else []

    for _ in range(n_positive_per_eq):
        sample_nr += 1
        vec, label = generate_positive_sample()
        evaluated = eval_formula(h, vec)
        if label is False and evaluated and (vec not in bad_nc):
            return (vec, sample_nr)
        if label is True and evaluated == False:
            return (vec, sample_nr)

    for _ in range(n_eq_samples):
        sample_nr += 1
        vec, label = generate_sample()
        evaluated = eval_formula(h, vec)
        if label is False and evaluated and (vec not in bad_nc):
            return (vec, sample_nr)
        if label is True and evaluated == False:
            return (vec, sample_nr)

    return (True, 0)