import unittest
from signal_detection import SignalDetection

class TestSignalDetection(unittest.TestCase):

    # Part 1: core SDT math tests
    def test_hit_rate(self):
        run1 = SignalDetection(80, 20, 15, 85)
        actual = run1.hit_rate()
        expected = 80/(80 + 20)
        self.assertEqual(actual, expected)

    def test_false_alarm_rate(self):
        run2 = SignalDetection(70, 30, 25, 75)
        actual = run2.false_alarm_rate()
        expected = 25/(75 + 25)
        self.assertEqual(actual, expected)

    def test_d_prime(self):
        run1 = SignalDetection(80, 20, 15, 85)
        actual = run1.d_prime()
        expected = 1.87805
        self.assertAlmostEqual(actual, expected, places=5)

    def test_criterion(self):
        run2 = SignalDetection(70, 30, 25, 75)
        actual = run2.criterion()
        expected = 0.07504
        self.assertAlmostEqual(actual, expected, places=5)

    # Part 2: input validation and object safety
    def test_input_type(self):
        with self.assertRaises(ValueError) as exception_context:
            SignalDetection(70, None, 25, 75)
        self.assertEqual(
            str(exception_context.exception),
            "All inputs must be numerical."
        )
    
    def test_input_value(self):
        with self.assertRaises(ValueError) as exception_context:
            SignalDetection(80, 20, -15, 85)
        self.assertEqual(
            str(exception_context.exception),
            "Inputs cannot be negative."
        )
    
    def test_addition_arguments(self):
        run1 = SignalDetection(80, 20, 15, 85)
        with self.assertRaises(TypeError) as exception_context:
            run1 + 5
        self.assertEqual(
            str(exception_context.exception),
            "Can only add 'SignalDetection'-type objects."
        )
    
    def test_subtraction_arguments(self):
        run2 = SignalDetection(70, 30, 25, 75)
        with self.assertRaises(TypeError) as exception_context:
            run2 - "definitely not an SD object..."
        self.assertEqual(
            str(exception_context.exception),
            "Can only subtract 'SignalDetection'-type objects."
        )
    
    def test_multiplication_arguments(self):
        run1 = SignalDetection(80, 20, 15, 85)
        with self.assertRaises(TypeError) as exception_context:
            run1 * None
        self.assertEqual(
            str(exception_context.exception),
            "Can only multiply using a numerical scalar."
        )
    
    # Part 3: operator behavior
    def test_addition(self):
        run1 = SignalDetection(80, 20, 15, 85)
        run2 = SignalDetection(70, 30, 25, 75)
        runs_added = run1 + run2
        actual = runs_added.d_prime()
        expected = 1.51611
        self.assertAlmostEqual(actual, expected, places=5)

    def test_subtraction(self):
        run1 = SignalDetection(80, 20, 15, 85)
        run2 = SignalDetection(70, 10, 5, 75)
        runs_subtracted = run1 - run2
        actual = runs_subtracted.d_prime()
        expected = 0
        self.assertEqual(actual, expected)
    
    def test_multiplication(self):
        run1 = SignalDetection(80, 20, 15, 85)
        runs_multiplied = run1 * 10
        actual = runs_multiplied.d_prime()
        expected = 1.87805
        self.assertAlmostEqual(actual, expected, places=5)
    
    def test_non_mutation(self):
        run1 = SignalDetection(80, 20, 15, 85)
        run2 = SignalDetection(70, 30, 25, 75)
        original_hits = run1.hits
        original_misses = run1.misses
        run_sum = run1 + run2
        self.assertEqual(run1.hits, original_hits)
        self.assertEqual(run1.misses, original_misses)

    # Part 4: plotting behavior
    def test_sdt_plot_axes(self):
        run1 = SignalDetection(80, 20, 15, 85)
        plot = SignalDetection.plot_sdt(run1)
        self.assertTrue(len(plot)>0)
        self.assertTrue(len(plot[1].lines)>0)
    
    def test_sdt_plot_labels(self):
        run1 = SignalDetection(80, 20, 15, 85)
        plot = run1.plot_sdt()
        self.assertEqual(plot[1].get_xlabel(), 'Internal response')
        self.assertEqual(plot[1].get_ylabel(), 'Probability density')
        self.assertEqual(plot[1].get_title(), 'SDT plot')

    def test_roc_plot_argument_type(self):
        with self.assertRaises(TypeError) as exception_context:
            SignalDetection.plot_roc("not a list...")
        self.assertEqual(
            str(exception_context.exception),
            "sdt_list must be either a list or a tuple."
        )
    
    def test_roc_plot_argument_values(self):
        run1 = SignalDetection(80, 20, 15, 85)
        with self.assertRaises(TypeError) as exception_context:
            SignalDetection.plot_roc([run1, "not an SD object"])
        self.assertEqual(
            str(exception_context.exception),
            "All elements of sdt_list must be 'SignalDetection'-type objects."
        )
    
    def test_roc_plot_endpoints(self):
        run1 = SignalDetection(80, 20, 15, 85)
        run2 = SignalDetection(70, 30, 25, 75)
        plot = SignalDetection.plot_roc([run1, run2])
        x_values = plot[1].lines[0].get_xdata()
        y_values = plot[1].lines[0].get_ydata()
        self.assertIn(0, x_values)
        self.assertIn(1, x_values)
        self.assertIn(0, y_values)
        self.assertIn(1, y_values)
    
    # The deliberate failure:
    def test_roc_plot_axes(self):
        # The ROC plot is "inverted" in original SD, 
        # i.e. false alarm should be on x-axis, 
        # and hit rate should be on y-axis.
        run1 = SignalDetection(80, 20, 15, 85)
        run2 = SignalDetection(70, 30, 25, 75)
        plot = SignalDetection.plot_roc([run1, run2])
        self.assertEqual(plot[1].get_xlabel(), 'False alarm rate')
        self.assertEqual(plot[1].get_ylabel(), 'Hit rate')