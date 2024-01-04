/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import PropTypes from "prop-types";
import { render } from "@testing-library/react";
import { ContextProvider } from "./utils/context";

export const renderWithContext = (ui, { providerProps, ...renderOptions } = {}) => {
    const Wrapper = ({ children }) => <ContextProvider {...providerProps}>{children}</ContextProvider>;

    Wrapper.propTypes = {
        children: PropTypes.object,
    };

    return render(ui, {
        wrapper: Wrapper,
        ...renderOptions,
    });
};
