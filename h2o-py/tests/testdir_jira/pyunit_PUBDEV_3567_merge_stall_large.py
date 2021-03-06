import sys
sys.path.insert(1,"../../")
import h2o
from tests import pyunit_utils

def pubdev_3567():
    train = h2o.import_file(pyunit_utils.locate("smalldata/jira/frameA2.csv"), header=1)
    test = h2o.import_file(pyunit_utils.locate("smalldata/jira/frameB2.csv"), header=1)
    mergedAns = h2o.import_file(pyunit_utils.locate("smalldata/jira/merged2.csv"), header=1)
    mergedAnsLeft = h2o.import_file(pyunit_utils.locate("smalldata/jira/merged2Left.csv"), header=1)
    mergedAnsRight = h2o.import_file(pyunit_utils.locate("smalldata/jira/merged2Right.csv"), header=1)
    merged = train.merge(test,by_x=["A"],by_y=["A"],method="auto") # default is radix
    mergedLeft = train.merge(test,by_x=["A"],by_y=["A"],all_x=True)
    mergedRight = train.merge(test,by_x=["A"],by_y=["A"],all_y=True)    # new feature

    pyunit_utils.compare_numeric_frames(mergedAnsRight, mergedRight, 0.0001, tol=1e-10)
    pyunit_utils.compare_numeric_frames(mergedAns, merged, 0.0001, tol=1e-10)
    pyunit_utils.compare_numeric_frames(mergedAnsLeft, mergedLeft, 0.0001, tol=1e-10)

if __name__ == "__main__":
    pyunit_utils.standalone_test(pubdev_3567)
else:
    pubdev_3567()
