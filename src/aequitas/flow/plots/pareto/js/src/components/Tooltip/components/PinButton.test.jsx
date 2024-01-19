/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import "@babel/polyfill";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom/extend-expect";

import { renderWithContext } from "~/testUtils";
import { mockAppState, mockTunerState } from "~/mocks/mockState";
import PinButton from "./PinButton";

const testAppState = { ...mockAppState, selectedModel: null, pinnedModel: { "model_id": 1 } };
const testTunerState = { ...mockTunerState, recommendedModel: { "model_id": 1 } };

test("Click on pin button pins model", async () => {
    const defaultProps = {
        model: { "model_id": 0 },
    };

    renderWithContext(<PinButton {...defaultProps} />, {
        providerProps: { appState: testAppState, tunerState: testTunerState },
    });

    expect(screen.getByText("Pin Model")).toBeInTheDocument();

    userEvent.click(screen.getByText("Pin Model"));
    await waitFor(() => expect(screen.getByText("Unpin Model")).toBeInTheDocument());
});

test("Click on unpin button clears model pin", async () => {
    const defaultProps = {
        model: { "model_id": 2 },
    };

    renderWithContext(<PinButton {...defaultProps} />, {
        providerProps: { appState: { ...testAppState, pinnedModel: { "model_id": 2 } }, tunerState: testTunerState },
    });

    expect(screen.getByText("Unpin Model")).toBeInTheDocument();

    userEvent.click(screen.getByText("Unpin Model"));
    await waitFor(() => expect(screen.getByText("Pin Model")).toBeInTheDocument());
});
