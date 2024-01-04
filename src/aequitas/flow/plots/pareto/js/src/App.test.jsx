/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import { screen, within } from "@testing-library/react";
import "@testing-library/jest-dom/extend-expect";

import { renderWithContext } from "~/testUtils";
import { mockAppState, mockTunerState } from "~/mocks/mockState";
import App from "~/App";

test("renders Tuner information", () => {
    renderWithContext(<App />, {
        providerProps: {
            appState: mockAppState,
            tunerState: {
                ...mockTunerState,
                alpha: "auto",
                tuner: "Fairband",
                optimizedFairnessMetric: "equal_opportunity",
                optimizedPerformanceMetric: "precision",
            },
        },
    });

    const { getByText } = within(screen.getByTestId("tuner-description"));

    expect(getByText(/Fairband/)).toBeInTheDocument();
    expect(getByText(/(α = auto)/)).toBeInTheDocument();
    expect(getByText(/# Precision/)).toBeInTheDocument();
    expect(getByText(/Equal Opportunity/)).toBeInTheDocument();
});
