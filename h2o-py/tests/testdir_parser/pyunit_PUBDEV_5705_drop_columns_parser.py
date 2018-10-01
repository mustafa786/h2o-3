from __future__ import print_function
import sys
sys.path.insert(1,"../../")
import h2o
from tests import pyunit_utils

# This test is to make sure that we have fixed the following JIRA properly using milsongs data:
# HEXDEV-497: Merged Gzip Files not read properly.
# I will import the original files and then the zip directory and compare them to see if they are the same.

def import_folder():
  file1 = h2o.import_file(pyunit_utils.locate("bigdata/laptop/airlinesBillion_7Columns_5GB.csv"), skipped_columns = [0,1])
  file2 = h2o.import_file(pyunit_utils.locate("bigdata/laptop/DGA.csv"))

  print("whoa")


if __name__ == "__main__":
  pyunit_utils.standalone_test(import_folder)
else:
  import_folder()
