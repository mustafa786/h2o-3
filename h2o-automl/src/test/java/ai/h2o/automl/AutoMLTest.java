package ai.h2o.automl;

import org.junit.BeforeClass;
import org.junit.Test;
import water.Key;
import water.fvec.Frame;

import java.util.Date;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

public class AutoMLTest extends TestUtil {

  @BeforeClass public static void setup() { stall_till_cloudsize(1); }

  @Test public void AirlinesTest() {
    AutoML aml=null;
    Frame fr=null;
    try {
      AutoMLBuildSpec autoMLBuildSpec = new AutoMLBuildSpec();
      fr = parse_test_file("./smalldata/airlines/allyears2k_headers.zip");
      autoMLBuildSpec.input_spec.training_frame = fr._key;
      autoMLBuildSpec.input_spec.response_column = "IsDepDelayed";
      autoMLBuildSpec.build_control.stopping_criteria.set_max_runtime_secs(5);
      autoMLBuildSpec.build_control.max_after_balance_size = 5.0f;
      autoMLBuildSpec.build_control.keep_cross_validation_models = false; //Prevent leaked keys from CV models
      autoMLBuildSpec.build_control.keep_cross_validation_predictions = false; //Prevent leaked keys from CV predictions

      aml = AutoML.makeAutoML(Key.<AutoML>make(), new Date(), autoMLBuildSpec);
      AutoML.startAutoML(aml);
      aml.get();

    } finally {
      // Cleanup
      if(aml!=null) aml.deleteWithChildren();
      if(fr != null) fr.remove();
    }
  }

  @Test public void KeepCrossValidationFoldAssignmentTest() {
    AutoML aml=null;
    Frame fr=null;
    try {
      AutoMLBuildSpec autoMLBuildSpec = new AutoMLBuildSpec();
      fr = parse_test_file("./smalldata/airlines/allyears2k_headers.zip");
      autoMLBuildSpec.input_spec.training_frame = fr._key;
      autoMLBuildSpec.input_spec.response_column = "IsDepDelayed";
      autoMLBuildSpec.build_control.stopping_criteria.set_max_runtime_secs(5);
      autoMLBuildSpec.build_control.keep_cross_validation_fold_assignment = true;

      aml = AutoML.makeAutoML(Key.<AutoML>make(), new Date(), autoMLBuildSpec);
      AutoML.startAutoML(aml);
      aml.get();

      assertTrue(aml.leader() !=null && aml.leader()._parms._keep_cross_validation_fold_assignment);
      assertTrue(aml.leader() !=null && aml.leader()._output._cross_validation_fold_assignment_frame_id != null);

    } finally {
      // cleanup
      if(aml!=null) aml.deleteWithChildren();
      if(fr != null) fr.remove();
    }
  }

  @Test public void testWorkPlan() {
    AutoML aml = null;
    Frame fr=null;
    try {
      AutoMLBuildSpec autoMLBuildSpec = new AutoMLBuildSpec();
      fr = parse_test_file("./smalldata/airlines/allyears2k_headers.zip");
      autoMLBuildSpec.input_spec.training_frame = fr._key;
      autoMLBuildSpec.input_spec.response_column = "IsDepDelayed";
      aml = new AutoML(Key.<AutoML>make(), new Date(), autoMLBuildSpec);

      AutoML.WorkAllocations workPlan = aml.planWork();

      int max_total_work = 1*10+3*20     //DL
                         + 2*10          //DRF
                         + 5*10+1*60     //GBM
                         + 1*20          //GLM
                         + 3*10+1*100    //XGBoost
                         + 2*15;         //SE
      assertEquals(workPlan.remainingWork(), max_total_work);

      autoMLBuildSpec.build_models.exclude_algos = new AutoML.Algo[] {AutoML.Algo.DeepLearning, AutoML.Algo.XGBoost, };
      workPlan = aml.planWork();

      assertEquals(workPlan.remainingWork(), max_total_work - (/*DL*/ 1*10+3*20 + /*XGB*/ 3*10+1*100));

    } finally {
      if (aml != null) aml.deleteWithChildren();
      if (fr != null) fr.remove();
    }
  }
}
