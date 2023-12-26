class ExampleClass:
    """Class Summary."""

    def __init__(self, arg1, arg2):
        """Purpose section description.

        Description section text.

        Parameters
        ----------
        arg1 : int
            First argument description.
        arg2 : float
            Second argument description.

        Returns
        -------
        bool

        """
        self.a = arg1
        "Attribute arg1 docstring."
        self.b = arg2
        "Attribute arg2 docstring."

    @property
    def square(self):
        """*(read-only)* Purpose section description.

        Returns
        -------
        int

        """
        return self.a * self.a
