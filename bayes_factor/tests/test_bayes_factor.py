import unittest
from bayes_factor.bayes_factor import BayesFactor

# 1) Input and state validation:
class TestBayesFactorConstruction(unittest.TestCase):

    def test_valid_construction(self):
        bf = BayesFactor(10, 3)
        self.assertEqual(bf.n, 10)
        self.assertEqual(bf.k, 3)
        self.assertAlmostEqual(bf.a, 0.4999)
        self.assertAlmostEqual(bf.b, 0.5001)

    def test_n_must_be_int_floatinput(self):
        with self.assertRaises(TypeError):
            BayesFactor(10.5, 3)

    def test_n_must_be_int_stringinput(self):
        with self.assertRaises(TypeError):
            BayesFactor("10", 3)

    def test_k_must_be_int_floatinput(self):
        with self.assertRaises(TypeError):
            BayesFactor(10, 3.5)

    def test_k_must_be_int_stringinput(self):
        with self.assertRaises(TypeError):
            BayesFactor(10, "3")

    def test_n_must_be_positive(self):
        with self.assertRaises(ValueError):
            BayesFactor(0, 0)

    def test_n_cannot_be_negative(self):
        with self.assertRaises(ValueError):
            BayesFactor(-1, 0)

    def test_k_cannot_be_negative(self):
        with self.assertRaises(ValueError):
            BayesFactor(10, -1)

    def test_k_cannot_exceed_n(self):
        with self.assertRaises(ValueError):
            BayesFactor(5, 6)

    def test_a_must_be_numerical(self):
        with self.assertRaises(TypeError):
            BayesFactor(10, 3, a="0.4")

    def test_b_must_be_numerical(self):
        with self.assertRaises(TypeError):
            BayesFactor(10, 3, b="0.6")

    def test_a_cannot_be_negative(self):
        with self.assertRaises(ValueError):
            BayesFactor(10, 3, a=-0.1)

    def test_b_cannot_exceed_one(self):
        with self.assertRaises(ValueError):
            BayesFactor(10, 3, b=1.1)

    def test_a_must_be_less_than_b(self):
        with self.assertRaises(ValueError):
            BayesFactor(10, 3, a=0.6, b=0.4)

    def test_a_equal_to_b_rejection(self):
        with self.assertRaises(ValueError):
            BayesFactor(10, 3, a=0.5, b=0.5)

    def test_k_equals_n_is_valid(self):
        bf = BayesFactor(5, 5)
        self.assertEqual(bf.k, 5)

    def test_k_equals_zero_is_valid(self):
        bf = BayesFactor(5, 0)
        self.assertEqual(bf.k, 0)


# 2) API behavior and return contracts:
class TestBayesFactorAPI(unittest.TestCase):

    def setUp(self):
        self.bf = BayesFactor(10, 3)

    def test_likelihood_is_callable(self):
        try:
            self.bf.likelihood(0.5)
        except AttributeError:
            self.fail("likelihood is not callable")

    def test_evidence_slab_is_callable(self):
        try:
            self.bf.evidence_slab()
        except AttributeError:
            self.fail("evidence_slab is not callable")

    def test_evidence_spike_is_callable(self):
        try:
            self.bf.evidence_spike()
        except AttributeError:
            self.fail("evidence_spike is not callable")

    def test_bayes_factor_is_callable(self):
        try:
            self.bf.bayes_factor()
        except AttributeError:
            self.fail("bayes_factor is not callable")

    def test_likelihood_returns_float(self):
        self.assertEqual(type(self.bf.likelihood(0.5)), float)

    def test_evidence_slab_returns_float(self):
        self.assertEqual(type(self.bf.evidence_slab()), float)

    def test_evidence_spike_returns_float(self):
        self.assertEqual(type(self.bf.evidence_spike()), float)

    def test_bayes_factor_returns_float(self):
        self.assertEqual(type(self.bf.bayes_factor()), float)


# 3) Mathematical consistency checks:
class TestLikelihood(unittest.TestCase):

    def setUp(self):
        self.bf = BayesFactor(10, 3)

    def test_likelihood_at_zero(self):
        self.assertEqual(self.bf.likelihood(0.0), 0.0)

    def test_likelihood_at_one(self):
        self.assertEqual(self.bf.likelihood(1.0), 0.0)

    def test_likelihood_is_not_negative(self):
        for theta in [0.0000000001, 0.3, 0.5, 0.7, 0.9999999]:
            self.assertGreaterEqual(self.bf.likelihood(theta), 0.0)

    def test_all_successes_peaks_at_one(self):
        bf = BayesFactor(10, 10)
        self.assertGreater(bf.likelihood(1.0), bf.likelihood(0.5))

    def test_all_failures_peaks_at_zero(self):
        bf = BayesFactor(10, 0)
        self.assertGreater(bf.likelihood(0.0), bf.likelihood(0.5))


class TestMathConsistency(unittest.TestCase):

    def setUp(self):
        self.bf = BayesFactor(10, 3)

    def test_evidence_slab_is_not_negative(self):
        self.assertGreaterEqual(self.bf.evidence_slab(), 0.0)

    def test_evidence_spike_is_not_negative(self):
        self.assertGreaterEqual(self.bf.evidence_spike(), 0.0)

    def test_bayes_factor_is_not_negative(self):
        self.assertGreaterEqual(self.bf.bayes_factor(), 0.0)

    def test_evidence_slab_closed_form(self):
        self.assertAlmostEqual(self.bf.evidence_slab(), 1 / (self.bf.n + 1), places=5)

    def test_evidence_slab_independent_of_k(self):
        for k in [0, 3, 7, 10]:
            bf = BayesFactor(10, k)
            self.assertAlmostEqual(bf.evidence_slab(), 1 / 11, places=5)

    def test_equal_priors_bayes_factor_gives_one(self):
        bf = BayesFactor(10, 3, a=0.0, b=1.0)
        self.assertAlmostEqual(bf.bayes_factor(), 1.0, places=5)

    def test_evidence_spike_near_mle_is_large(self):
        bf_near = BayesFactor(10, 3, a=0.29, b=0.31)
        bf_far = BayesFactor(10, 3, a=0.7, b=0.72)
        self.assertGreater(bf_near.evidence_spike(), bf_far.evidence_spike())

    def test_bayes_factor_favors_spike_near_mle(self):
        bf = BayesFactor(10, 3, a=0.29, b=0.31)
        self.assertGreater(bf.bayes_factor(), 1.0)

    def test_bayes_factor_disfavors_spike_far_from_mle(self):
        bf = BayesFactor(10, 3, a=0.7, b=0.72)
        self.assertLess(bf.bayes_factor(), 1.0)

    def test_bayes_factor_no_successes(self):
        bf = BayesFactor(10, 0)
        self.assertGreaterEqual(bf.bayes_factor(), 0.0)

    def test_bayes_factor_all_successes(self):
        bf = BayesFactor(10, 10)
        self.assertGreaterEqual(bf.bayes_factor(), 0.0)

    def test_bayes_factor_spike_at_lower_boundary(self):
        bf = BayesFactor(10, 3, a=0.0, b=0.0001)
        self.assertGreaterEqual(bf.bayes_factor(), 0.0)

    def test_bayes_factor_spike_at_upper_boundary(self):
        bf = BayesFactor(10, 3, a=0.9999, b=1.0)
        self.assertGreaterEqual(bf.bayes_factor(), 0.0)


# 4) Error behavior:
class TestErrorBehavior(unittest.TestCase):

    def setUp(self):
        self.bf = BayesFactor(10, 3)

    def test_likelihood_theta_too_large(self):
        with self.assertRaises(ValueError):
            self.bf.likelihood(1.5)

    def test_likelihood_theta_negative(self):
        with self.assertRaises(ValueError):
            self.bf.likelihood(-0.1)

    def test_likelihood_theta_not_numerical(self):
        with self.assertRaises(TypeError):
            self.bf.likelihood("0.5")

    def test_error_message_n_type(self):
        with self.assertRaises(TypeError) as T_error:
            BayesFactor("ten", 3)
        self.assertIn("n must be an integer", str(T_error.exception))

    def test_error_message_k_exceeds_n(self):
        with self.assertRaises(ValueError) as V_error:
            BayesFactor(5, 6)
        self.assertIn("k cannot exceed n", str(V_error.exception))

    def test_error_message_n_positive(self):
        with self.assertRaises(ValueError) as V_error:
            BayesFactor(0, 0)
        self.assertIn("n must be positive", str(V_error.exception))

    def test_error_message_theta_type(self):
        with self.assertRaises(TypeError) as T_error:
            self.bf.likelihood("0.5")
        self.assertIn("theta must be numerical", str(T_error.exception))

    def test_error_message_theta_range(self):
        with self.assertRaises(ValueError) as V_error:
            self.bf.likelihood(1.5)
        self.assertIn("theta must be in [0, 1]", str(V_error.exception))


# 5) The intentionally failing test:
class TestIntentionallyFailing(unittest.TestCase):
    def test_bayes_factor_large_n(self):
        # It so happens that math.comb(n, k) overflows for large n...
        bf = BayesFactor(10000, 5000)
        self.assertGreaterEqual(bf.bayes_factor(), 0.0)