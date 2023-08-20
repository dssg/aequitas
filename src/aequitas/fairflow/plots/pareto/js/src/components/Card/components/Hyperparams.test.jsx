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

import { renderWithContext } from "~/testUtils";
import { mockAppState, mockTunerState } from "~/mocks/mockState";
import Hyperparams from "~/components/Card/components/Hyperparams";

test("Hyperparams displays params names and values for single model", () => {
    const testAppState = {
        ...mockAppState,
        pinnedModel: {
            "hyperparams": {
                "classpath": "test_model_type",
                "param1": 0.553,
                "param2": "gini",
                "param3": 3570,
                "param4": true,
            },
        },
        selectedModel: null,
    };

    renderWithContext(<Hyperparams />, { providerProps: { appState: testAppState, tunerState: mockTunerState } });
    expect(screen.getByText(/param1/)).toBeInTheDocument();
    expect(screen.getByText(/0.553/)).toBeInTheDocument();
    expect(screen.getByText(/param2/)).toBeInTheDocument();
    expect(screen.getByText(/gini/)).toBeInTheDocument();
    expect(screen.getByText(/param3/)).toBeInTheDocument();
    expect(screen.getByText(/3.57k/)).toBeInTheDocument();
    expect(screen.getByText(/param4/)).toBeInTheDocument();
    expect(screen.getByText(/true/)).toBeInTheDocument();
});

test("Hyperparams displays N/A for hyperparams not common to both pinned and selected model", () => {
    const testAppState = {
        ...mockAppState,
        pinnedModel: {
            "hyperparams": {
                "classpath": "test_model_type",
                "param1": 0.553,
            },
        },
        selectedModel: {
            "hyperparams": {
                "classpath": "test_model_type_alt",
                "param2": "value",
            },
        },
    };

    renderWithContext(<Hyperparams />, { providerProps: { appState: testAppState, tunerState: mockTunerState } });
    expect(screen.getByText(/param1/)).toBeInTheDocument();
    expect(screen.getByText(/0.55/)).toBeInTheDocument();
    expect(screen.getByText(/param2/)).toBeInTheDocument();
    expect(screen.getByText(/value/)).toBeInTheDocument();
    expect(screen.getAllByText(/N\/A/)).toHaveLength(2);
});
