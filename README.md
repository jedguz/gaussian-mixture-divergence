# gaussian-mixture-divergence

HSD-GM/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── hsdgm/
│       ├── __init__.py
│       ├── instances.py        # default_instance, random_simplex
│       ├── math_utils.py       # phi, torch_cdf, g_eps (np/torch), gprime_eps
│       ├── constraints.py      # project, check_feasible
│       ├── estimators.py       # sample_H_mixture, sample_H (soft hinge), estimate_H_mc
│       ├── bounds/
│       │   ├── __init__.py
│       │   ├── constant.py     # constant_bound, collapsed_bound
│       │   ├── trivial_1d.py   # trivial_spacing_exact, exact_mixture_1d
│       │   ├── multitangent_socp.py  # multitangent_socp (+ make_envelopes)
│       │   ├── gram_socp.py    # ub_gram_socp, ub_gram_socp_cheb, _postprocess
│       │   ├── sdp_coord.py    # ub_sdp (your exp_envelope_knots helper here)
│       │   └── analytic.py     # ub_renyi
│       ├── nonconvex/
│       │   └── exact_t.py      # exact_t_solver
│       └── pgd/
│           ├── pgd_pair.py     # pgd_U_pair_corrected
│           └── pgd_full.py     # pgd_H_full_corrected
├── scripts/
│   ├── smoke.py                # runs smoketest() and prints results
│   ├── compare_bounds.py       # grid_search + CSV dump
│   └── collapsed_suite.py      # sanity_suite_square
├── tests/
│   ├── test_constraints.py
│   ├── test_estimators.py
│   ├── test_bounds_smoke.py
│   └── test_collapsed.py
└── notebooks/
    └── playground.ipynb        # optional, for ad-hoc experiments
