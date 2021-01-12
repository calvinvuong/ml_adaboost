import argparse
import os

# Take in the single path parameter and break it into a filename and path.
class ProcessPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        file_location, file_name = os.path.split(values)
        setattr(namespace, self.dest, file_location)
        setattr(namespace, "name", file_name)

# Take in a 0 or 1 and convert it into a boolean.
class ProcessTFAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values == 1)

# Create the parser for naive bayes input arguments
# return: The parser which will parse naive baye's input.
def parser_naive_args():
    parser = argparse.ArgumentParser(description="Utilize a Naive Bayes classifier on sample data")
    
    shared_options(parser)
    
    parser.add_argument(
                "num_bins", 
                action="store", 
                type=int,
                help="The number of desired bins for discretizing continuous atributes.",
                default=5
            )

    parser.add_argument(
                "m_val", 
                action="store", 
                type=float,
                help="The value of m, for m-estimate",
                default=.1
            )
    return parser

# Create the parser for logistic regression input arguments
# return: The parser which will parse logistic regression input.
def parser_logreg_args():
    parser = argparse.ArgumentParser(description="Utilize a Logistic Regression classifier on sample data")
    shared_options(parser)
    parser.add_argument(
                "lambd", 
                action="store", 
                type=nonnegative_float,
                help="Decay hyper parameter"
            )
    
    parser.add_argument(
                "correlate", 
                action="store", 
                nargs='?', 
                type=int, 
                default=False
            )
    
    return parser

# Create the parser for the boosting parameters
# return: The parse which can be used to parse arguments.
def parser_boost_args():
    parser = argparse.ArgumentParser(description="Utilize a boostead learner for various base learners")
    shared_options(parser)
    parser.add_argument(
        "algorithm",
        action="store",
        type=str,
        help="The algorithm to use with the boosted learner",
        default="logreg"
    )
    parser.add_argument(
        "iterations",
        action="store",
        type=int,
        help="The number of iterations as the upper bound in boosting (1 for base learner)",
        default=30
    )
    parser.add_argument(
        "extension",
        action="store",
        nargs='?',
        type=int,
        help="0 if standard, 1 if extension",
        default=0
    )
    parser.add_argument(
        "print_folds",
        action="store",
        nargs='?',
        type=int,
        help="0 if don't print fold merics, 1 if print fold metrics",
        default=0
    )
    
    return parser


# Apply options which are shared accross both classifiers:
# parser: The parser we want to add the arguments to.
def shared_options(parser):
    parser.add_argument(
                "path", 
                action=ProcessPathAction,
                help="The path (including name) of desired sample data."
            )

    parser.add_argument(
                "full_sample", 
                action=ProcessTFAction, 
                type=int,
                help="Whether to use the full example data.",
                default=True
            )

# Return float of the value if it is non negative
# val: The input argument
# return: Non negative typed argument
def nonnegative_float(val):
    val_float = float(val)
    if val_float < 0:
        raise argparse.ArgumentTypeError("%s is an invalid non negative float value" % val_float)
    return val_float

# Helper method to parse the given arguments with the provided parser.
def parse_arguments(args, parser):
    return parser.parse_args(args)
