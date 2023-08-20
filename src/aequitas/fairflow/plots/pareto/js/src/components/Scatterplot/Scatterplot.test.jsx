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
import Scatterplot from "~/components/Scatterplot";

test("Renders Pareto line if it is visible", () => {
    renderWithContext(<Scatterplot />, { providerProps: { appState: mockAppState, tunerState: mockTunerState } });

    expect(screen.getByTestId("pareto-line")).toBeInTheDocument();
});
