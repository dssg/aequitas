/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import { Line } from "@visx/shape";

import labels from "~/enums/labels";
import { useAppDispatch, useAppState } from "~/utils/context";
import { ACTIONS } from "~/utils/reducer";

import "./ParetoItem.scss";

const PARETO_LINE_LENGTH = 20; // px
const PARETO_LINE_STOKE = 3; // px

function ParetoItem() {
    const { isParetoVisible, isParetoDisabled } = useAppState();
    const dispatch = useAppDispatch();

    const handleCheckBoxChange = () => {
        dispatch({ type: ACTIONS.TOGGLE_PARETO_VISIBILITY });
    };

    return (
        <>
            <input
                type="checkbox"
                name="pareto"
                id="pareto"
                onChange={handleCheckBoxChange}
                checked={isParetoVisible}
                disabled={isParetoDisabled}
            />
            <label htmlFor="pareto">
                <span className="checkbox-text-label">{labels.legend.PARETO}</span>
            </label>
            <svg width={PARETO_LINE_LENGTH} height={PARETO_LINE_STOKE} className="legend-item-symbol">
                <Line
                    from={{ x: 0, y: PARETO_LINE_STOKE / 2 }}
                    to={{ x: PARETO_LINE_LENGTH, y: PARETO_LINE_STOKE / 2 }}
                    className="pareto-line"
                />
            </svg>
        </>
    );
}

export default ParetoItem;
