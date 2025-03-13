from pyqubo import Array, Constraint, Placeholder, solve_qubo

class QBooster():
    def __init__(self, y_train, ys_pred):
        self.num_clf = ys_pred.shape[0]
        # define binary variables
        self.Ws = Array.create("weight", shape = self.num_clf, vartype="BINARY")
        # set the Hyperparameter size of the normalization term with PyQUBO's Placeholder
        self.param_lamda = Placeholder("norm")
        #　set the Hamiltonian of the combination of weak classifiers
        self.H_clf = sum( [ (1/self.num_clf * sum([W*C for W, C in zip(self.Ws, y_clf)])- y_true)**2 for y_true, y_clf in zip(y_train, ys_pred.T)
        ])
        # set the Hamiltonian of the normalization term as a constraint
        self.H_norm = Constraint(sum([W for W in self.Ws]), "norm")
        
        self.H = self.H_clf + self.H_norm * self.param_lamda
        self.model = self.H.compile()

    def to_qubo(self, norm_param=1):
        self.feed_dict = {'norm': norm_param}
        return self.model.to_qubo(feed_dict=self.feed_dict)