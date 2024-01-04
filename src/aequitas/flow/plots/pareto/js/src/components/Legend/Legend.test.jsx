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
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { renderWithContext } from "~/testUtils";
import { mockAppState, mockTunerState } from "~/mocks/mockState";
import App from "~/App";

test("Unchecking Pareto Frontier checkbox removes Pareto Line in Scatterplot", async () => {
    renderWithContext(<App />, { providerProps: { appState: mockAppState, tunerState: mockTunerState } });

    expect(screen.getByText("Pareto Frontier")).toBeInTheDocument();

    const paretoCheckbox = screen.getByRole("checkbox");
    expect(paretoCheckbox).toBeChecked();

    userEvent.click(paretoCheckbox);
    await waitFor(() => expect(screen.queryByTestId("pareto-line")).not.toBeInTheDocument());
});
