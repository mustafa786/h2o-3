package hex.tree.xgboost;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.gbm.GBTree;
import biz.k11i.xgboost.gbm.GradBooster;
import biz.k11i.xgboost.tree.RegTree;
import biz.k11i.xgboost.tree.RegTreeImpl;
import hex.DataInfo;
import hex.genmodel.algos.tree.SharedTreeGraph;
import hex.genmodel.algos.tree.SharedTreeNode;
import hex.genmodel.algos.tree.SharedTreeSubgraph;
import ml.dmlc.xgboost4j.java.XGBoostModelInfo;
import water.H2O;
import water.fvec.Frame;
import water.util.Log;

import java.io.ByteArrayInputStream;
import java.io.IOException;

public class XGBoostTreeConverter {

    public static SharedTreeSubgraph convertXGBoostTree(final XGBoostModel xgBoostModel, final int treeNumber, final int treeClass) {
        final XGBoostModelInfo xgBoostModelInfo = xgBoostModel.model_info();
        GradBooster booster = null;
        try {
            booster = new Predictor(new ByteArrayInputStream(xgBoostModelInfo._boosterBytes)).getBooster();
        } catch (IOException e) {
            Log.err(e);
            H2O.fail(e.getMessage());
        }

        if (!(booster instanceof GBTree)) {
            throw new IllegalArgumentException(String.format("Given XGBoost model is not backed by a tree-based booster. Booster class is %d",
                    booster.getClass().getCanonicalName()));
        }

        final RegTree[][] groupedTrees = ((GBTree) booster).getGroupedTrees();
        if (treeClass >= groupedTrees.length) {
            throw new IllegalArgumentException("Given XGBoost model does not have given class"); //Todo: better info - print at least number of existing classes, ideal situation would be to print the tring
        }

        final RegTree[] treesInGroup = groupedTrees[treeClass];

        if (treeNumber >= treesInGroup.length) {
            throw new IllegalArgumentException("There is no such tree number for given class"); // Todo: better info - same as above
        }

        final RegTreeImpl.Node[] treeNodes = treesInGroup[treeNumber].getNodes();
        assert treeNodes.length >= 1;

        SharedTreeGraph sharedTreeGraph = new SharedTreeGraph();
        final SharedTreeSubgraph sharedTreeSubgraph = sharedTreeGraph.makeSubgraph(xgBoostModel._output._training_metrics._description);
        treeNodes[0].split_index();

        final FeatureProperties featureProperties = assembleFeatureNames(xgBoostModel.model_info()._dataInfoKey.get()); // XGBoost's usage of one-hot encoding assumed
        constructSubgraph(treeNodes, sharedTreeSubgraph.makeRootNode(), 0, sharedTreeSubgraph, featureProperties, true); // Root node is at index 0
        return sharedTreeSubgraph;

    }

    private static void constructSubgraph(final RegTreeImpl.Node[] xgBoostNodes, final SharedTreeNode sharedTreeNode,
                                          final int nodeIndex, final SharedTreeSubgraph sharedTreeSubgraph,
                                          final FeatureProperties featureProperties, boolean inclusiveNA) {
        final RegTreeImpl.Node xgBoostNode = xgBoostNodes[nodeIndex];
        // Not testing for NaNs, as SharedTreeNode uses NaNs as default values.
        //No domain set, as the structure mimics XGBoost's tree, which is numeric-only
        if (featureProperties._oneHotEncoded[xgBoostNode.split_index()]) {
            //Shared tree model uses < to the left and >= to the right. Transforiming one-hot encoded categoricals
            // from 0 to 1 makes it fit the current split description logic
            sharedTreeNode.setSplitValue(1.0F);
        } else {
            sharedTreeNode.setSplitValue(xgBoostNode.getSplitCondition());
        }
        sharedTreeNode.setPredValue(xgBoostNode.getLeafValue());
        sharedTreeNode.setCol(xgBoostNode.split_index(), featureProperties._names[xgBoostNode.split_index()]);
        sharedTreeNode.setInclusiveNa(inclusiveNA);
        sharedTreeNode.setNodeNumber(nodeIndex);

        if (xgBoostNode.getLeftChildIndex() != -1) {
            constructSubgraph(xgBoostNodes, sharedTreeSubgraph.makeLeftChildNode(sharedTreeNode),
                    xgBoostNode.getLeftChildIndex(), sharedTreeSubgraph, featureProperties, xgBoostNode.default_left());
        }

        if (xgBoostNode.getRightChildIndex() != -1) {
            constructSubgraph(xgBoostNodes, sharedTreeSubgraph.makeRightChildNode(sharedTreeNode),
                    xgBoostNode.getRightChildIndex(), sharedTreeSubgraph, featureProperties, !xgBoostNode.default_left());
        }
    }


    private static FeatureProperties assembleFeatureNames(final DataInfo di) {
        String[] coefnames = di.coefNames();
        assert (coefnames.length == di.fullN());
        final Frame frame = di._adaptedFrame;
        int numCatCols = di._catOffsets[di._catOffsets.length - 1];

        String[] featureNames = new String[di.fullN()];
        boolean[] oneHotEncoded = new boolean[di.fullN()];
        for (int i = 0; i < di.fullN(); ++i) {
            featureNames[i] = coefnames[i];
            if (i < numCatCols) {
                oneHotEncoded[i] = true;
            }
        }

        return new FeatureProperties(featureNames, oneHotEncoded);
    }

    private static class FeatureProperties {
        private String[] _names;
        private boolean[] _oneHotEncoded;

        public FeatureProperties(String[] names, boolean[] oneHotEncoded) {
            _names = names;
            _oneHotEncoded = oneHotEncoded;
        }
    }


}
