/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import { isNumber } from "lodash";

import labels from "~/enums/labels";
import { formatNumber, formatLargeInteger } from "~/utils/formatters";
import { useAppState } from "~/utils/context";

export default function Hyperparams() {
    const { pinnedModel, selectedModel } = useAppState();
    const models = [pinnedModel, selectedModel].filter(Boolean);

    const modelsHyperparams = models.map((model) =>
        Object.keys(model.hyperparams).filter((value) => value !== "classpath"),
    );
    const hyperparams = [...new Set([].concat(...modelsHyperparams))];

    const formatCellValue = (value) => {
        if (isNumber(value)) {
            return value > 999 ? formatLargeInteger(value) : formatNumber(value);
        }
        return value.toString();
    };

    return (
        <>
            <tr className="title">
                <td>{labels.HYPERPARAMS}</td>
            </tr>
            {hyperparams.map((param) => {
                return (
                    <tr className={"metrics-row"} key={`table-hyper-${param}`}>
                        <td>{param}</td>
                        {models.map((model, index) => {
                            const isPinned = index === 0;
                            let cellClassName = "value";
                            let cellValue;

                            if (Object.prototype.hasOwnProperty.call(model.hyperparams, param)) {
                                cellValue = formatCellValue(model.hyperparams[param]);
                            } else {
                                cellValue = "N/A";
                                cellClassName += " missing";
                            }

                            return (
                                <td className={cellClassName} key={`col-${isPinned}-${param}`}>
                                    {cellValue}
                                </td>
                            );
                        })}
                    </tr>
                );
            })}
        </>
    );
}
