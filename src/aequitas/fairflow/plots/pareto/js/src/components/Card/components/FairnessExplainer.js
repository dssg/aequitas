/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import Tippy from "@tippyjs/react";

import { RiInformationLine as HelperIcon } from "react-icons/ri";
import { formatFairnessLabel } from "~/utils/formatters";

import labels from "~/enums/labels";

export default function FairnessExplainer() {
    return (
        <Tippy
            content={
                <div>
                    {["predictive_equality", "equal_opportunity"].map((metric) => (
                        <p key={`helper-text-${metric}`}>
                            <span className="bold">{formatFairnessLabel(metric)}: </span>
                            {labels.HELPERS[metric]}
                        </p>
                    ))}
                </div>
            }
            placement="top"
            className="tooltip-overflow card"
            appendTo={document.getElementsByClassName("fair-app")[0]}
            trigger="click"
        >
            <span className="helper-icon">
                <HelperIcon />
            </span>
        </Tippy>
    );
}
