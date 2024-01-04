/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React, { useEffect, useRef, useState } from "react";

import Header from "./components/Header";
import Metrics from "./components/Metrics";
import Hyperparams from "./components/Hyperparams";
import labels from "~/enums/labels";
import { formatFairnessLabel, formatPerformanceLabel } from "~/utils/formatters";
import { useTunerState } from "~/utils/context";

import "./Card.scss";

const spacer = <tr className="spacer" />;

function Card() {
    const { fairnessMetrics, performanceMetrics, optimizedFairnessMetric, optimizedPerformanceMetric } =
        useTunerState();
    const tableWrapperRef = useRef();
    const [showScrollArrow, setShowScrollArrow] = useState(false);
    const [showScrollBar, setShowScrollBar] = useState(false);

    const updateScrollArrow = () => {
        const tableNode = tableWrapperRef.current;

        setShowScrollBar(tableNode.scrollTop > 0);

        /* We check if there is a section of the card which is overflowing not on the current card viewport. 
        When tableNode.scrollHeight = tableNode.scrollTop + tableNode.clientHeight we are the bottom of
        the scroll. We use 10 instead of 0 as in some displays this value doesn't seem to reach exactly 0.  */
        const isScrollable = tableNode.scrollHeight - tableNode.scrollTop - tableNode.clientHeight > 10;

        setShowScrollArrow(isScrollable);
    };

    useEffect(() => {
        updateScrollArrow();
    }, [tableWrapperRef]);

    return (
        <div className="card card-models-stats">
            <div
                className={`stats-table-wrapper ${!showScrollBar ? "scrollbar-hidden" : ""}`}
                ref={tableWrapperRef}
                onScroll={updateScrollArrow}
            >
                <table className="stats-table">
                    <thead>
                        <Header />
                    </thead>
                    <tbody>
                        <Metrics
                            title={labels.FAIRNESS}
                            metrics={fairnessMetrics}
                            optimizedMetric={optimizedFairnessMetric}
                            metricLabelFormatter={formatFairnessLabel}
                            isFairness
                        />
                        {spacer}
                        <Metrics
                            title={labels.PERFORMANCE}
                            metrics={performanceMetrics}
                            optimizedMetric={optimizedPerformanceMetric}
                            metricLabelFormatter={formatPerformanceLabel}
                        />
                        {spacer}
                        <Hyperparams />
                    </tbody>
                </table>
            </div>
            {showScrollArrow ? <div className="scroll-icon arrow down grey" /> : null}
        </div>
    );
}

export default Card;
