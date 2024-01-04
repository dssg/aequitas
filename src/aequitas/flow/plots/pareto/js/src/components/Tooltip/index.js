/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import PropTypes from "prop-types";
import React from "react";

import PinButton from "./components/PinButton";
import {
    formatFairnessLabel,
    formatPerformanceLabel,
    formatPercentage,
    formatPercentagePoint,
} from "~/utils/formatters";
import { useAppState } from "~/utils/context";
import { id } from "~/utils/models";
import labels from "~/enums/labels";

import "./Tooltip.scss";

function Tooltip({ fairnessMetric, performanceMetric, model, isRecommended, isPinned, ...props }) {
    const { pinnedModel } = useAppState();

    const renderDifference = (metric) => {
        const difference = model[metric] - pinnedModel[metric];

        if (difference === 0) {
            return (
                <div className={`row`}>
                    <span>{labels.EQUAL}</span>
                </div>
            );
        }
        const isHigher = difference > 0;
        const color = isHigher ? "green" : "orange";
        const arrowDirection = isHigher ? "up" : "down";

        return (
            <div className={`row ${color}`}>
                <span data-testid={`arrow ${arrowDirection}`} className={`arrow ${arrowDirection} ${color}`} />
                <span>{formatPercentagePoint(Math.abs(difference), false)} </span>
            </div>
        );
    };

    return (
        <div className="tooltip card" {...props}>
            <div className="row">
                <h1>Model {id(model)}</h1>
                {isRecommended ? <h2 className="blue label">{labels.RECOMMENDED}</h2> : null}
            </div>
            <div className={`content ${isPinned ? "pinned" : null}`}>
                <span>{formatFairnessLabel(fairnessMetric)}:</span>
                <span className="bold">{formatPercentage(model[fairnessMetric])}</span>
                {!isPinned ? renderDifference(fairnessMetric) : null}
                <span>{formatPerformanceLabel(performanceMetric)}:</span>
                <span className="bold">{formatPercentage(model[performanceMetric])}</span>
                {!isPinned ? renderDifference(performanceMetric) : null}
            </div>
            {!(isPinned && isRecommended) ? <PinButton model={model} /> : null}
        </div>
    );
}

Tooltip.propTypes = {
    fairnessMetric: PropTypes.string.isRequired,
    performanceMetric: PropTypes.string.isRequired,
    model: PropTypes.object.isRequired,
    isRecommended: PropTypes.bool.isRequired,
    isPinned: PropTypes.bool.isRequired,
};

export default Tooltip;
