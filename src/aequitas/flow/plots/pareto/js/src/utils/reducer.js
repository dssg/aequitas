/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import { isNull } from "lodash";

import { id } from "~/utils/models";

export const ACTIONS = {
    SELECT_MODEL: "SELECT_MODEL",
    PIN_MODEL: "PIN_MODEL",
    CLEAR_MODEL_SELECTION: "CLEAR_MODEL_SELECTION",
    CLEAR_MODEL_PIN: "CLEAR_MODEL_PIN",
    ENABLE_PARETO: "ENABLE_PARETO",
    DISABLE_PARETO: "DISABLE_PARETO",
    TOGGLE_PARETO_VISIBILITY: "TOGGLE_PARETO_VISIBILITY",
};

export const initialState = {
    selectedModel: null,
    isParetoDisabled: false,
    isParetoVisible: true,
};

/* eslint-disable complexity */
export const AppReducer = (state, action) => {
    switch (action.type) {
        case ACTIONS.SELECT_MODEL:
            return {
                ...state,
                selectedModel: id(action.model) !== id(state.pinnedModel) ? action.model : null,
            };
        case ACTIONS.PIN_MODEL: {
            let selectedModel = state.selectedModel;

            if (!isNull(selectedModel) && id(action.model) === id(selectedModel)) {
                selectedModel = null;
            }

            return {
                ...state,
                selectedModel,
                pinnedModel: action.model,
            };
        }
        case ACTIONS.CLEAR_MODEL_SELECTION:
            return { ...state, selectedModel: null };
        case ACTIONS.CLEAR_MODEL_PIN:
            return { ...state, pinnedModel: action.recommendedModel };
        case ACTIONS.ENABLE_PARETO:
            return { ...state, isParetoDisabled: false, isParetoVisible: true };
        case ACTIONS.DISABLE_PARETO:
            return { ...state, isParetoDisabled: true, isParetoVisible: false };
        case ACTIONS.TOGGLE_PARETO_VISIBILITY:
            return { ...state, isParetoVisible: !state.isParetoVisible };
        default:
            return state;
    }
};
