/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React, { createContext, useContext, useReducer } from "react";
import PropTypes from "prop-types";
import { AppReducer } from "./reducer";

const AppStateContext = createContext();
const AppDispatchContext = createContext();
const TunerStateContext = createContext();

export const ContextProvider = ({ children, tunerState, appState }) => {
    const [state, dispatch] = useReducer(AppReducer, appState);

    return (
        <TunerStateContext.Provider value={tunerState}>
            <AppStateContext.Provider value={state}>
                <AppDispatchContext.Provider value={dispatch}>{children}</AppDispatchContext.Provider>
            </AppStateContext.Provider>
        </TunerStateContext.Provider>
    );
};

ContextProvider.propTypes = {
    children: PropTypes.object.isRequired,
    appState: PropTypes.shape({
        pinnedModel: PropTypes.object.isRequired,
        isParetoDisabled: PropTypes.bool.isRequired,
        isParetoVisible: PropTypes.bool.isRequired,
        selectedModel: PropTypes.object,
    }).isRequired,
    tunerState: PropTypes.shape({
        models: PropTypes.arrayOf(PropTypes.object).isRequired,
        recommendedModel: PropTypes.object.isRequired,
        fairnessMetrics: PropTypes.arrayOf(PropTypes.string).isRequired,
        performanceMetrics: PropTypes.arrayOf(PropTypes.string).isRequired,
        optimizedFairnessMetric: PropTypes.string.isRequired,
        optimizedPerformanceMetric: PropTypes.string.isRequired,
        tuner: PropTypes.string.isRequired,
        alpha: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    }).isRequired,
};

export const useTunerState = () => {
    return useContext(TunerStateContext);
};

export const useAppState = () => {
    return useContext(AppStateContext);
};

export const useAppDispatch = () => {
    return useContext(AppDispatchContext);
};
