package ai.h2o.automl;

// if we need to make the Algo list dynamic, we should just turn this enum into a class...
// NOTE: make sure that this is in sync with the exclude option in AutoMLBuildSpecV99
public enum Algo implements AutoML.algo {
  GLM,
  DRF,
  GBM,
  DeepLearning,
  StackedEnsemble,
  XGBoost,
  ;

  String urlName() {
    return this.name().toLowerCase();
  }
}
