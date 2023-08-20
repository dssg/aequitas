/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import "@babel/polyfill";
import "@testing-library/jest-dom/extend-expect";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { renderWithContext } from "~/testUtils";
import { mockAppState, mockTunerState } from "~/mocks/mockState";
import App from "~/App";
import Header from "~/components/Card/components/Header";

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
    "precision": 0.865,
    "predictive_equality": 0.456,
    "recall": 0.678,
    "is_pareto": false,
    "hyperparams": {
        "classpath": "sklearn.neural_network.MLPClassifier",
        "param4": 2,
        "param5": 332,
    },
};

test("Header for recommended model displays model info and recommended tag", () => {
    const testAppState = {
        ...mockAppState,
        pinnedModel: MODEL_1,
        selectedModel: null,
    };
    const testTunerState = {
        ...mockTunerState,
        recommendedModel: MODEL_1,
    };

    renderWithContext(<Header />, { providerProps: { appState: testAppState, tunerState: testTunerState } });
    expect(screen.getByText(/Model 0/)).toBeInTheDocument();
    expect(screen.getByText(/Recommended/)).toBeInTheDocument();
    expect(screen.getByText(/RandomForestClassifier/)).toBeInTheDocument();
});

test("Header shows model info for pinned and selected model", () => {
    const testAppState = {
        ...mockAppState,
        pinnedModel: MODEL_1,
        selectedModel: MODEL_2,
    };
    const testTunerState = {
        ...mockTunerState,
        recommendedModel: MODEL_1,
    };

    renderWithContext(<Header />, { providerProps: { appState: testAppState, tunerState: testTunerState } });
    expect(screen.getByText(/Model 0/)).toBeInTheDocument();
    expect(screen.getByText(/Recommended/)).toBeInTheDocument();
    expect(screen.getByText(/RandomForestClassifier/)).toBeInTheDocument();

    expect(screen.getByText(/Model 1/)).toBeInTheDocument();
    expect(screen.getByText(/MLPClassifier/)).toBeInTheDocument();
});

test("Click on model point shows its model information on Card", async () => {
    const testAppState = {
        ...mockAppState,
        pinnedModel: MODEL_1,
        selectedModel: null,
    };
    const testTunerState = {
        ...mockTunerState,
        recommendedModel: MODEL_1,
    };

    renderWithContext(<App />, {
        providerProps: {
            appState: testAppState,
            tunerState: testTunerState,
        },
    });

    userEvent.click(screen.getByTestId("point-1"));

    const modelID = await screen.findByText(/Model 1/);
    const modelType = screen.getByText(/MLPClassifier/);

    expect(modelID).toBeInTheDocument();
    expect(modelType).toBeInTheDocument();
});
