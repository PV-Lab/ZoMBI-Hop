"""Third-party measured datasets packaged as zhbench objectives.

Each module here owns one external dataset: how to fetch it, what it actually
contains, and how to turn it into the plain dict ``objectives.build`` needs
(``name, dim, fn, true_optima, true_values, maximize, domain, meta``). They return
a dict rather than an :class:`~..objectives.Objective` so nothing in this package
has to import the registry that would import it back.

Data lands in the gitignored ``data/`` tree and is regenerable from scratch:

    python -m benchmarks.zhbench.datasets.oer --fetch

Nothing large is ever committed.
"""
