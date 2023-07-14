/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";

import labels from "~/enums/labels";
import { useTunerState } from "~/utils/context";
import { formatFairnessLabel, formatPerformanceLabel } from "~/utils/formatters";

import Card from "~/components/Card";
import Scatterplot from "~/components/Scatterplot";
import Legend from "~/components/Legend";

import "./App.scss";

function App() {
    const { tuner, alpha, optimizedFairnessMetric, optimizedPerformanceMetric } = useTunerState();

    return (
        <div className="fair-app">
            <h1>{labels.TITLE}</h1>
            <p data-testid="tuner-description">
                Optimized for
                <span className="bold"> {formatPerformanceLabel(optimizedPerformanceMetric)}</span> and
                <span className="bold"> {formatFairnessLabel(optimizedFairnessMetric)} </span>
                with
                <span className="bold"> {tuner}</span> (α = {alpha})
            </p>
            <div className="visualization">
                <Scatterplot />
                <Card />
            </div>
            <Legend />
        </div>
    );
}

export default App;