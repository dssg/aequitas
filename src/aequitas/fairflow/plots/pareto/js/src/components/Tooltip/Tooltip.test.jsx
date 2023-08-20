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
import Tooltip from "~/components/Tooltip";
import App from "~/App";

const testState = {
    ...mockAppState,
    pinnedModel: { "model_id": 0, "precision": 0.8765, "equal_opportunity": 0.6543 },
};

test("Tooltip displays selected metric values for hovered model point", () => {
    const defaultProps = {
        fairnessMetric: "equal_opportunity",
        performanceMetric: "precision",
        model: { "model_id": 1, "precision": 0.3456, "equal_opportunity": 0.789 },
        isRecommended: false,
        isPinned: false,
    };

    renderWithContext(<Tooltip {...defaultProps} />, {
        providerProps: { appState: testState, tunerState: mockTunerState },
    });

    expect(screen.getByText("Model 1")).toBeInTheDocument();
    expect(screen.getByText(/Precision/)).toBeInTheDocument();
    expect(screen.getByText(/Equal Opportunity/)).toBeInTheDocument();
    expect(screen.getByText("34.6%")).toBeInTheDocument();
    expect(screen.getByText("78.9%")).toBeInTheDocument();
});

test("Tooltip displays selected metric tradeoffs for hovered model point", () => {
    const defaultProps = {
        fairnessMetric: "equal_opportunity",
        performanceMetric: "precision",
        model: { "model_id": 1, "precision": 0.3456, "equal_opportunity": 0.789 },
        isRecommended: false,
        isPinned: false,
    };

    renderWithContext(<Tooltip {...defaultProps} />, {
        providerProps: { appState: testState, tunerState: mockTunerState },
    });

    expect(screen.getByText("53.1 pp")).toBeInTheDocument();
    expect(screen.getByTestId("arrow down")).toBeInTheDocument();
    expect(screen.getByText("13.5 pp")).toBeInTheDocument();
    expect(screen.getByTestId("arrow up")).toBeInTheDocument();
});

test("Hover on model point shows its Tooltip", async () => {
    renderWithContext(<App />, {
        providerProps: { appState: { ...mockAppState, selectedModel: null }, tunerState: mockTunerState },
    });

    userEvent.hover(screen.getByTestId("point-1"));
    const modelID = await screen.findByText(/Model 1/);
    expect(modelID).toBeInTheDocument();
});
