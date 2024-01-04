/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React, { useMemo } from "react";
import PropTypes from "prop-types";
import Tippy from "@tippyjs/react";
import { GlyphCircle, GlyphStar } from "@visx/glyph";
import { isNull } from "lodash";

import { useAppDispatch, useAppState, useTunerState } from "~/utils/context";
import { ACTIONS } from "~/utils/reducer";
import { sortModels, id, isPareto } from "~/utils/models";
import chartSettings from "~/constants/scatterplot";

import Tooltip from "~/components/Tooltip";

function Points({ fairnessMetric, performanceMetric, xScale, yScale }) {
    const { models, recommendedModel } = useTunerState();
    const { pinnedModel, selectedModel } = useAppState();
    const dispatch = useAppDispatch();

    const sortedModels = useMemo(
        () => sortModels(models, recommendedModel, pinnedModel, selectedModel),
        [models, recommendedModel, pinnedModel, selectedModel],
    );

    const handleClickOnPoint = (model) => {
        dispatch({ type: ACTIONS.SELECT_MODEL, model });
    };

    return (
        <g>
            {sortedModels.map((model) => {
                const isRecommended = id(model) === id(recommendedModel);
                const isSelected = !isNull(selectedModel) && id(model) === id(selectedModel);
                const isPinned = id(model) === id(pinnedModel);
                const borderClassName = `pinned-border ${isRecommended ? "recommended" : ""}`;

                let pointClassName = "point interactive";

                pointClassName += isRecommended ? " recommended" : "";
                pointClassName += isSelected || isPinned ? " selected" : "";
                pointClassName += isPareto(model) ? " pareto" : "";

                const Point = isRecommended ? GlyphStar : GlyphCircle;

                return (
                    <Tippy
                        key={id(model)}
                        content={
                            <Tooltip
                                fairnessMetric={fairnessMetric}
                                performanceMetric={performanceMetric}
                                model={model}
                                isRecommended={isRecommended}
                                isPinned={isPinned}
                                data-testid={`tooltip-${id(model)}`}
                            />
                        }
                        interactive
                        offset={[0, 0]}
                        delay={350}
                        appendTo={document.getElementsByClassName("scatterplot-wrapper")[0]}
                    >
                        <g>
                            {isPinned && !isRecommended ? (
                                <Point
                                    left={xScale(model[performanceMetric])}
                                    top={yScale(model[fairnessMetric])}
                                    size={chartSettings.pointSize + 100}
                                    className={borderClassName}
                                />
                            ) : null}
                            <Point
                                left={xScale(model[performanceMetric])}
                                top={yScale(model[fairnessMetric])}
                                size={chartSettings.pointSize}
                                className={pointClassName}
                                onClick={() => handleClickOnPoint(model)}
                                data-testid={`point-${id(model)}`}
                            />
                        </g>
                    </Tippy>
                );
            })}
        </g>
    );
}

Points.propTypes = {
    fairnessMetric: PropTypes.string.isRequired,
    performanceMetric: PropTypes.string.isRequired,
    xScale: PropTypes.func.isRequired,
    yScale: PropTypes.func.isRequired,
};

export default Points;
