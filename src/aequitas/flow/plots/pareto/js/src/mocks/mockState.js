/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

const MODEL_1 = {
    "model_id": 0,
    "equal_opportunity": 0.987,
    "precision": 0.765,
    "predictive_equality": 0.789,
    "recall": 0.456,
    "is_pareto": true,
    "hyperparams": {
        "classpath": "sklearn.ensemble.RandomForestClassifier",
        "param1": 0.54,
        "param2": 20,
        "param3": "left",
    },
};

const MODEL_2 = {
    "model_id": 1,
    "equal_opportunity": 0.374,
    "precision": 0.078,
    "predictive_equality": 0.456,
    "recall": 0.678,
    "is_pareto": false,
    "hyperparams": {
        "classpath": "sklearn.neural_network.MLPClassifier",
        "param4": 2,
        "param5": 332,
    },
};

const MODEL_3 = {
    "model_id": 2,
    "equal_opportunity": 0.685,
    "precision": 0.799,
    "predictive_equality": 0.234,
    "recall": 0.123,
    "is_pareto": true,
    "hyperparams": {
        "classpath": "lightgbm.LGBMClassifier",
        "param6": 0.54,
        "param7": 20,
        "param8": "right",
    },
};

export const mockAppState = {
    pinnedModel: MODEL_2,
    selectedModel: null,
    isParetoVisible: true,
    isParetoDisabled: false,
};

export const mockTunerState = {
    models: [MODEL_1, MODEL_2, MODEL_3],
    recommendedModel: MODEL_1,
    optimizedFairnessMetric: "equal_opportunity",
    optimizedPerformanceMetric: "precision",
    fairnessMetrics: ["equal_opportunity", "predictive_equality"],
    performanceMetrics: ["precision", "recall"],
    tuner: "Fairband",
    alpha: "auto",
};
