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

test("Points for models have correct styling", () => {
    renderWithContext(<Scatterplot />, { providerProps: { appState: mockAppState, tunerState: mockTunerState } });

    /* The visx-glyph-circle and visx-glyph-star class names como from visx and can be used to 
    identify the shape of the glyph in use. https://github.com/airbnb/visx/tree/master/packages/visx-glyph */
    expect(screen.getByTestId("point-1")).toHaveClass("visx-glyph-circle point interactive");
    expect(screen.getByTestId("point-0")).toHaveClass("visx-glyph-star point interactive recommended pareto");
    expect(screen.getByTestId("point-2")).toHaveClass("visx-glyph-circle point interactive pareto");
});
