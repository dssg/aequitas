/**
 * The copyright of this file belongs to Feedzai. The file cannot be reproduced in whole or in part,
 * stored in a retrieval system, transmitted in any form, or by any means electronic, mechanical, photocopying,
 * or otherwise, without the prior permission of the owner.
 *
 * © 2021 Feedzai, Strictly Confidential
 */

import React from "react";
import Tippy from "@tippyjs/react";
import { RiPushpin2Line as PinIcon } from "react-icons/ri";
import { isNull } from "lodash";

import { ACTIONS } from "~/utils/reducer";
import { useAppState, useTunerState, useAppDispatch } from "~/utils/context";
import { id } from "~/utils/models";

export default function Header() {
    const { pinnedModel, selectedModel } = useAppState();
    const { recommendedModel } = useTunerState();
    const dispatch = useAppDispatch();

    const handleUnselect = () => dispatch({ type: ACTIONS.CLEAR_MODEL_SELECTION });

    return (
        <tr>
            <td />
            {[pinnedModel, selectedModel].filter(Boolean).map((model, index) => {
                const isRecommended = id(model) === id(recommendedModel);
                const isPinned = index === 0;

                let modelType = model.hyperparams.classpath;

                /* Model types usually have long names such as "sklearn.ensemble.RandomForestClassifier". 
                The most meaningful part to show is the last substring "RandomForestClassifier". We will
                check if the model type string follows this pattern, and if it does, we will capture
                the last substring after the "." to display */
                const shorterModelTypeMatches = model.hyperparams.classpath.match(/(?<=\.)[a-zA-Z]+$/);

                if (!isNull(shorterModelTypeMatches)) {
                    modelType = shorterModelTypeMatches[0];
                }

                return (
                    <td key={`col-${id(model)}-${isPinned}-header`}>
                        {!isPinned ? (
                            <button onClick={handleUnselect} className="unselect-button">
                                Unselect
                            </button>
                        ) : null}
                        {isRecommended ? <h2 className="blue label">Recommended</h2> : null}
                        <h1>
                            {isPinned ? <PinIcon /> : null}
                            Model {id(model)}
                        </h1>
                        <Tippy
                            content={model.hyperparams.classpath}
                            placement="bottom"
                            className="tooltip-overflow card"
                            appendTo={document.getElementsByClassName("fair-app")[0]}
                        >
                            <h2 className="badge">{modelType}</h2>
                        </Tippy>
                    </td>
                );
            })}
        </tr>
    );
}
