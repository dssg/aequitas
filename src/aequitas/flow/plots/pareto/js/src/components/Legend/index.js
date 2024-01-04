/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";

import ModelItem from "./components/ModelItem";
import ParetoItem from "./components/ParetoItem";
import labels from "~/enums/labels";

import "./Legend.scss";

function Legend() {
    return (
        <div className="legend">
            <div className="legend-row">
                <ModelItem labelText={labels.legend.CANDIDATE_MODELS} pointClassName="point" />
                <ModelItem
                    labelText={labels.legend.RECOMMENDED_MODEL}
                    pointClassName="point recommended pareto"
                    isRecommended
                />
            </div>
            <div className="legend-row">
                <ParetoItem />
            </div>
        </div>
    );
}

export default Legend;
