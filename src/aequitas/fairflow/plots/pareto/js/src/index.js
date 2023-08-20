/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import ReactDOM from "react-dom";
import { select } from "d3-selection";
import { ErrorBoundary } from "react-error-boundary";

/* Load Roboto font */
import "@fontsource/roboto/300.css";
import "@fontsource/roboto/400.css";
import "@fontsource/roboto/700.css";

import App from "./App";
import { ContextProvider } from "./utils/context";
import { initialState } from "./utils/reducer";
import ErrorFallback from "./components/ErrorFallback";

export function render(divId, payload) {
    const models = JSON.parse(payload.models);

    ReactDOM.render(
        <ErrorBoundary FallbackComponent={ErrorFallback}>
            <ContextProvider
                tunerState={{
                    models,
                    recommendedModel: payload.recommended_model,
                    fairnessMetrics: payload["fairness_metrics"],
                    performanceMetrics: payload["performance_metrics"],
                    optimizedFairnessMetric: payload["optimized_fairness_metric"],
                    optimizedPerformanceMetric: payload["optimized_performance_metric"],
                    tuner: payload["tuner_type"],
                    alpha: payload.alpha,
                }}
                appState={{ ...initialState, pinnedModel: payload.recommended_model }}
            >
                <App />
            </ContextProvider>
        </ErrorBoundary>,
        select(divId).node(),
    );
}
