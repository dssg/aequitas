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
import { mockAppState, mockTunerState } from "~/mocks/mockState";

import DivergingBar from "~/components/Card/components/DivergingBar";

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
const testAppState = { ...mockAppState, selectedModel: MODEL_2, pinnedModel: MODEL_1 };

test("Diverging Bar shows correct and formatted difference value and has orange color when negative", async () => {
    const defaultProps = {
        metric: "equal_opportunity",
    };
    renderWithContext(<DivergingBar {...defaultProps} />, {
        providerProps: { appState: testAppState, tunerState: mockTunerState },
    });
    expect(screen.getByText(/−61.3 pp/)).toBeInTheDocument();
    expect(screen.getByTestId("diverging-rect")).toHaveClass("orange");
});

test("Diverging Bar shows correct and formatted difference value and has green color when positive", async () => {
    const defaultProps = {
        metric: "precision",
    };
    renderWithContext(<DivergingBar {...defaultProps} />, {
        providerProps: { appState: testAppState, tunerState: mockTunerState },
    });
    expect(screen.getByText(/\+10.0 pp/)).toBeInTheDocument();
    expect(screen.getByTestId("diverging-rect")).toHaveClass("green");
});
