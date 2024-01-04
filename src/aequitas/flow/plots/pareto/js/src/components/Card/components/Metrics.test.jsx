/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import "@babel/polyfill";
import { screen } from "@testing-library/react";
import "@testing-library/jest-dom/extend-expect";

import { renderWithContext } from "~/testUtils";
import { formatPerformanceLabel } from "~/utils/formatters";
import { mockAppState, mockTunerState } from "~/mocks/mockState";
import Metrics from "~/components/Card/components/Metrics";

const MODEL_1 = {
    "model_id": 0,
    "equal_opportunity": 0.987,
    "precision": 0.765,
    "predictive_equality": 0.789,
    "recall": 0.456,
    "is_pareto": true,
    "hyperparams": {
        "classpath": "Random Forest",
        "param1": 0.54,
        "param2": 20,
        "param3": "left",
    },
};

const MODEL_2 = {
    "model_id": 1,
    "equal_opportunity": 0.374,
    "precision": 0.865,
    "predictive_equality": 0.456,
    "recall": 0.678,
    "is_pareto": false,
    "hyperparams": {
        "classpath": "Neural Network",
        "param4": 2,
        "param5": 332,
    },
};

const testAppState = {
    ...mockAppState,
    selectedModel: MODEL_2,
    pinnedModel: MODEL_1,
};

test("Metric section shows values correctly formatted", () => {
    const defaultProps = {
        metrics: ["precision", "recall"],
        optimizedMetric: "precision",
        title: "Performance",
        metricLabelFormatter: formatPerformanceLabel,
    };

    renderWithContext(<Metrics {...defaultProps} />, {
        providerProps: { appState: testAppState, tunerState: mockTunerState },
    });
    expect(screen.getByText(/Precision/)).toBeInTheDocument();
    expect(screen.getByText(/Recall/)).toBeInTheDocument();
    expect(screen.getByText(/76.5%/)).toBeInTheDocument();
    expect(screen.getByText(/45.6%/)).toBeInTheDocument();
    expect(screen.getByText(/86.5%/)).toBeInTheDocument();
    expect(screen.getByText(/67.8%/)).toBeInTheDocument();
});
