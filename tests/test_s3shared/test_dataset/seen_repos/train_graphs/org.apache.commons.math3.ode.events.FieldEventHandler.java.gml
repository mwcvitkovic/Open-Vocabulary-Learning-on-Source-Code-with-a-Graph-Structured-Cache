<?xml version="1.0" ?><graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.1/graphml.xsd"><key id="reference" for="node" attr.name="reference" attr.type="string"></key><key id="identifier" for="node" attr.name="identifier" attr.type="string"></key><key id="modifier" for="node" attr.name="modifier" attr.type="string"></key><key id="text" for="node" attr.name="text" attr.type="string"></key><key id="type" for="node" attr.name="type" attr.type="string"></key><key id="parentType" for="node" attr.name="parentType" attr.type="string"></key><key id="type" for="edge" attr.name="type" attr.type="string"></key><graph id="G" edgedefault="directed"><node id="44"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * Compute the value of the switching function.
 *
 * &lt;p&gt;The discrete events are generated when the sign of this
 * switching function changes. The integrator will take care to change
 * the stepsize in such a way these events occur exactly at step boundaries.
 * The switching function must be continuous in its roots neighborhood
 * (but not necessarily smooth), as the integrator will need to find its
 * roots to locate precisely the events.&lt;/p&gt;
 * &lt;p&gt;Also note that the integrator expect that once an event has occurred,
 * the sign of the switching function at the start of the next step (i.e.
 * just after the event) is the opposite of the sign just before the event.
 * This consistency between the steps &lt;string&gt;must&lt;/strong&gt; be preserved,
 * otherwise {@link org.apache.commons.math3.exception.NoBracketingException
 * exceptions} related to root not being bracketed will occur.&lt;/p&gt;
 * &lt;p&gt;This need for consistency is sometimes tricky to achieve. A typical
 * example is using an event to model a ball bouncing on the floor. The first
 * idea to represent this would be to have {@code g(t) = h(t)} where h is the
 * height above the floor at time {@code t}. When {@code g(t)} reaches 0, the
 * ball is on the floor, so it should bounce and the typical way to do this is
 * to reverse its vertical velocity. However, this would mean that before the
 * event {@code g(t)} was decreasing from positive values to 0, and after the
 * event {@code g(t)} would be increasing from 0 to positive values again.
 * Consistency is broken here! The solution here is to have {@code g(t) = sign
 * * h(t)}, where sign is a variable with initial value set to {@code +1}. Each
 * time {@link #eventOccurred(FieldODEStateAndDerivative, boolean) eventOccurred}
 * method is called, {@code sign} is reset to {@code -sign}. This allows the
 * {@code g(t)} function to remain continuous (and even smooth) even across events,
 * despite {@code h(t)} is not. Basically, the event is used to &lt;em&gt;fold&lt;/em&gt;
 * {@code h(t)} at bounce points, and {@code sign} is used to &lt;em&gt;unfold&lt;/em&gt; it
 * back, so the solvers sees a {@code g(t)} function which behaves smoothly even
 * across events.&lt;/p&gt;
 *
 * @param state current value of the independent &lt;i&gt;time&lt;/i&gt; variable, state vector
 * and derivative
 * @return value of the g switching function
 */
T g(FieldODEStateAndDerivative&lt;T&gt; state);</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="45"><data key="reference">userDefinedMethodName</data><data key="identifier">g</data><data key="text">g</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="46"><data key="reference"></data><data key="modifier"></data><data key="text">FieldODEStateAndDerivative&lt;T&gt; state</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="47"><data key="reference"></data><data key="text">FieldODEStateAndDerivative&lt;T&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="48"><data key="reference">nonQualifiedClassName</data><data key="identifier">FieldODEStateAndDerivative</data><data key="text">FieldODEStateAndDerivative</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="49"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="50"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="51"><data key="reference">org.apache.commons.math3.ode.FieldODEStateAndDerivative</data><data key="identifier">state</data><data key="text">state</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="52"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="53"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="10"><data key="reference"></data><data key="identifier">RealFieldElement</data><data key="text">org.apache.commons.math3.RealFieldElement</data><data key="type">Name</data><data key="parentType">ImportDeclaration</data></node><node id="54"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * Handle an event and choose what to do next.
 *
 * &lt;p&gt;This method is called when the integrator has accepted a step
 * ending exactly on a sign change of the function, just &lt;em&gt;before&lt;/em&gt;
 * the step handler itself is called (see below for scheduling). It
 * allows the user to update his internal data to acknowledge the fact
 * the event has been handled (for example setting a flag in the {@link
 * org.apache.commons.math3.ode.FirstOrderDifferentialEquations
 * differential equations} to switch the derivatives computation in
 * case of discontinuity), or to direct the integrator to either stop
 * or continue integration, possibly with a reset state or derivatives.&lt;/p&gt;
 *
 * &lt;ul&gt;
 *   &lt;li&gt;if {@link Action#STOP} is returned, the step handler will be called
 *   with the &lt;code&gt;isLast&lt;/code&gt; flag of the {@link
 *   org.apache.commons.math3.ode.sampling.StepHandler#handleStep handleStep}
 *   method set to true and the integration will be stopped,&lt;/li&gt;
 *   &lt;li&gt;if {@link Action#RESET_STATE} is returned, the {@link #resetState
 *   resetState} method will be called once the step handler has
 *   finished its task, and the integrator will also recompute the
 *   derivatives,&lt;/li&gt;
 *   &lt;li&gt;if {@link Action#RESET_DERIVATIVES} is returned, the integrator
 *   will recompute the derivatives,
 *   &lt;li&gt;if {@link Action#CONTINUE} is returned, no specific action will
 *   be taken (apart from having called this method) and integration
 *   will continue.&lt;/li&gt;
 * &lt;/ul&gt;
 *
 * &lt;p&gt;The scheduling between this method and the {@link
 * org.apache.commons.math3.ode.sampling.FieldStepHandler FieldStepHandler} method {@link
 * org.apache.commons.math3.ode.sampling.FieldStepHandler#handleStep(
 * org.apache.commons.math3.ode.sampling.FieldStepInterpolator, boolean)
 * handleStep(interpolator, isLast)} is to call this method first and
 * &lt;code&gt;handleStep&lt;/code&gt; afterwards. This scheduling allows the integrator to
 * pass &lt;code&gt;true&lt;/code&gt; as the &lt;code&gt;isLast&lt;/code&gt; parameter to the step
 * handler to make it aware the step will be the last one if this method
 * returns {@link Action#STOP}. As the interpolator may be used to navigate back
 * throughout the last step, user code called by this method and user
 * code called by step handlers may experience apparently out of order values
 * of the independent time variable. As an example, if the same user object
 * implements both this {@link FieldEventHandler FieldEventHandler} interface and the
 * {@link org.apache.commons.math3.ode.sampling.FieldStepHandler FieldStepHandler}
 * interface, a &lt;em&gt;forward&lt;/em&gt; integration may call its
 * {code eventOccurred} method with t = 10 first and call its
 * {code handleStep} method with t = 9 afterwards. Such out of order
 * calls are limited to the size of the integration step for {@link
 * org.apache.commons.math3.ode.sampling.FieldStepHandler variable step handlers}.&lt;/p&gt;
 *
 * @param state current value of the independent &lt;i&gt;time&lt;/i&gt; variable, state vector
 * and derivative
 * @param increasing if true, the value of the switching function increases
 * when times increases around event (note that increase is measured with respect
 * to physical time, not with respect to integration which may go backward in time)
 * @return indication of what the integrator should do next, this
 * value must be one of {@link Action#STOP}, {@link Action#RESET_STATE},
 * {@link Action#RESET_DERIVATIVES} or {@link Action#CONTINUE}
 */
Action eventOccurred(FieldODEStateAndDerivative&lt;T&gt; state, boolean increasing);</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="11"><data key="reference"></data><data key="identifier">math3</data><data key="text">org.apache.commons.math3</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="55"><data key="reference">userDefinedMethodName</data><data key="identifier">eventOccurred</data><data key="text">eventOccurred</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="12"><data key="reference"></data><data key="identifier">commons</data><data key="text">org.apache.commons</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="56"><data key="reference"></data><data key="modifier"></data><data key="text">FieldODEStateAndDerivative&lt;T&gt; state</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="13"><data key="reference"></data><data key="identifier">apache</data><data key="text">org.apache</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="57"><data key="reference"></data><data key="text">FieldODEStateAndDerivative&lt;T&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="14"><data key="reference"></data><data key="identifier">org</data><data key="text">org</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="58"><data key="reference">nonQualifiedClassName</data><data key="identifier">FieldODEStateAndDerivative</data><data key="text">FieldODEStateAndDerivative</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="15"><data key="reference"></data><data key="text">import org.apache.commons.math3.ode.FieldODEState;
</data><data key="type">ImportDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="59"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="16"><data key="reference"></data><data key="identifier">FieldODEState</data><data key="text">org.apache.commons.math3.ode.FieldODEState</data><data key="type">Name</data><data key="parentType">ImportDeclaration</data></node><node id="17"><data key="reference"></data><data key="identifier">ode</data><data key="text">org.apache.commons.math3.ode</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="18"><data key="reference"></data><data key="identifier">math3</data><data key="text">org.apache.commons.math3</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="19"><data key="reference"></data><data key="identifier">commons</data><data key="text">org.apache.commons</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="1"><data key="reference"></data><data key="text">/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.commons.math3.ode.events;

import org.apache.commons.math3.RealFieldElement;
import org.apache.commons.math3.ode.FieldODEState;
import org.apache.commons.math3.ode.FieldODEStateAndDerivative;

/**
 * This interface represents a handler for discrete events triggered
 * during ODE integration.
 *
 * &lt;p&gt;Some events can be triggered at discrete times as an ODE problem
 * is solved. This occurs for example when the integration process
 * should be stopped as some state is reached (G-stop facility) when the
 * precise date is unknown a priori, or when the derivatives have
 * discontinuities, or simply when the user wants to monitor some
 * states boundaries crossings.
 * &lt;/p&gt;
 *
 * &lt;p&gt;These events are defined as occurring when a &lt;code&gt;g&lt;/code&gt;
 * switching function sign changes.&lt;/p&gt;
 *
 * &lt;p&gt;Since events are only problem-dependent and are triggered by the
 * independent &lt;i&gt;time&lt;/i&gt; variable and the state vector, they can
 * occur at virtually any time, unknown in advance. The integrators will
 * take care to avoid sign changes inside the steps, they will reduce
 * the step size when such an event is detected in order to put this
 * event exactly at the end of the current step. This guarantees that
 * step interpolation (which always has a one step scope) is relevant
 * even in presence of discontinuities. This is independent from the
 * stepsize control provided by integrators that monitor the local
 * error (this event handling feature is available for all integrators,
 * including fixed step ones).&lt;/p&gt;
 *
 * @param &lt;T&gt; the type of the field elements
 * @since 3.6
 */
public interface FieldEventHandler&lt;T extends RealFieldElement&lt;T&gt;&gt; {

    /**
     * Initialize event handler at the start of an ODE integration.
     * &lt;p&gt;
     * This method is called once at the start of the integration. It
     * may be used by the event handler to initialize some internal data
     * if needed.
     * &lt;/p&gt;
     * @param initialState initial time, state vector and derivative
     * @param finalTime target time for the integration
     */
    void init(FieldODEStateAndDerivative&lt;T&gt; initialState, T finalTime);

    /**
     * Compute the value of the switching function.
     *
     * &lt;p&gt;The discrete events are generated when the sign of this
     * switching function changes. The integrator will take care to change
     * the stepsize in such a way these events occur exactly at step boundaries.
     * The switching function must be continuous in its roots neighborhood
     * (but not necessarily smooth), as the integrator will need to find its
     * roots to locate precisely the events.&lt;/p&gt;
     * &lt;p&gt;Also note that the integrator expect that once an event has occurred,
     * the sign of the switching function at the start of the next step (i.e.
     * just after the event) is the opposite of the sign just before the event.
     * This consistency between the steps &lt;string&gt;must&lt;/strong&gt; be preserved,
     * otherwise {@link org.apache.commons.math3.exception.NoBracketingException
     * exceptions} related to root not being bracketed will occur.&lt;/p&gt;
     * &lt;p&gt;This need for consistency is sometimes tricky to achieve. A typical
     * example is using an event to model a ball bouncing on the floor. The first
     * idea to represent this would be to have {@code g(t) = h(t)} where h is the
     * height above the floor at time {@code t}. When {@code g(t)} reaches 0, the
     * ball is on the floor, so it should bounce and the typical way to do this is
     * to reverse its vertical velocity. However, this would mean that before the
     * event {@code g(t)} was decreasing from positive values to 0, and after the
     * event {@code g(t)} would be increasing from 0 to positive values again.
     * Consistency is broken here! The solution here is to have {@code g(t) = sign
     * * h(t)}, where sign is a variable with initial value set to {@code +1}. Each
     * time {@link #eventOccurred(FieldODEStateAndDerivative, boolean) eventOccurred}
     * method is called, {@code sign} is reset to {@code -sign}. This allows the
     * {@code g(t)} function to remain continuous (and even smooth) even across events,
     * despite {@code h(t)} is not. Basically, the event is used to &lt;em&gt;fold&lt;/em&gt;
     * {@code h(t)} at bounce points, and {@code sign} is used to &lt;em&gt;unfold&lt;/em&gt; it
     * back, so the solvers sees a {@code g(t)} function which behaves smoothly even
     * across events.&lt;/p&gt;
     *
     * @param state current value of the independent &lt;i&gt;time&lt;/i&gt; variable, state vector
     * and derivative
     * @return value of the g switching function
     */
    T g(FieldODEStateAndDerivative&lt;T&gt; state);

    /**
     * Handle an event and choose what to do next.
     *
     * &lt;p&gt;This method is called when the integrator has accepted a step
     * ending exactly on a sign change of the function, just &lt;em&gt;before&lt;/em&gt;
     * the step handler itself is called (see below for scheduling). It
     * allows the user to update his internal data to acknowledge the fact
     * the event has been handled (for example setting a flag in the {@link
     * org.apache.commons.math3.ode.FirstOrderDifferentialEquations
     * differential equations} to switch the derivatives computation in
     * case of discontinuity), or to direct the integrator to either stop
     * or continue integration, possibly with a reset state or derivatives.&lt;/p&gt;
     *
     * &lt;ul&gt;
     *   &lt;li&gt;if {@link Action#STOP} is returned, the step handler will be called
     *   with the &lt;code&gt;isLast&lt;/code&gt; flag of the {@link
     *   org.apache.commons.math3.ode.sampling.StepHandler#handleStep handleStep}
     *   method set to true and the integration will be stopped,&lt;/li&gt;
     *   &lt;li&gt;if {@link Action#RESET_STATE} is returned, the {@link #resetState
     *   resetState} method will be called once the step handler has
     *   finished its task, and the integrator will also recompute the
     *   derivatives,&lt;/li&gt;
     *   &lt;li&gt;if {@link Action#RESET_DERIVATIVES} is returned, the integrator
     *   will recompute the derivatives,
     *   &lt;li&gt;if {@link Action#CONTINUE} is returned, no specific action will
     *   be taken (apart from having called this method) and integration
     *   will continue.&lt;/li&gt;
     * &lt;/ul&gt;
     *
     * &lt;p&gt;The scheduling between this method and the {@link
     * org.apache.commons.math3.ode.sampling.FieldStepHandler FieldStepHandler} method {@link
     * org.apache.commons.math3.ode.sampling.FieldStepHandler#handleStep(
     * org.apache.commons.math3.ode.sampling.FieldStepInterpolator, boolean)
     * handleStep(interpolator, isLast)} is to call this method first and
     * &lt;code&gt;handleStep&lt;/code&gt; afterwards. This scheduling allows the integrator to
     * pass &lt;code&gt;true&lt;/code&gt; as the &lt;code&gt;isLast&lt;/code&gt; parameter to the step
     * handler to make it aware the step will be the last one if this method
     * returns {@link Action#STOP}. As the interpolator may be used to navigate back
     * throughout the last step, user code called by this method and user
     * code called by step handlers may experience apparently out of order values
     * of the independent time variable. As an example, if the same user object
     * implements both this {@link FieldEventHandler FieldEventHandler} interface and the
     * {@link org.apache.commons.math3.ode.sampling.FieldStepHandler FieldStepHandler}
     * interface, a &lt;em&gt;forward&lt;/em&gt; integration may call its
     * {code eventOccurred} method with t = 10 first and call its
     * {code handleStep} method with t = 9 afterwards. Such out of order
     * calls are limited to the size of the integration step for {@link
     * org.apache.commons.math3.ode.sampling.FieldStepHandler variable step handlers}.&lt;/p&gt;
     *
     * @param state current value of the independent &lt;i&gt;time&lt;/i&gt; variable, state vector
     * and derivative
     * @param increasing if true, the value of the switching function increases
     * when times increases around event (note that increase is measured with respect
     * to physical time, not with respect to integration which may go backward in time)
     * @return indication of what the integrator should do next, this
     * value must be one of {@link Action#STOP}, {@link Action#RESET_STATE},
     * {@link Action#RESET_DERIVATIVES} or {@link Action#CONTINUE}
     */
    Action eventOccurred(FieldODEStateAndDerivative&lt;T&gt; state, boolean increasing);

    /**
     * Reset the state prior to continue the integration.
     *
     * &lt;p&gt;This method is called after the step handler has returned and
     * before the next step is started, but only when {@link
     * #eventOccurred(FieldODEStateAndDerivative, boolean) eventOccurred} has itself
     * returned the {@link Action#RESET_STATE} indicator. It allows the user to reset
     * the state vector for the next step, without perturbing the step handler of the
     * finishing step. If the {@link #eventOccurred(FieldODEStateAndDerivative, boolean)
     * eventOccurred} never returns the {@link Action#RESET_STATE} indicator, this
     * function will never be called, and it is safe to leave its body empty.&lt;/p&gt;
     * @param state current value of the independent &lt;i&gt;time&lt;/i&gt; variable, state vector
     * and derivative
     * @return reset state (note that it does not include the derivatives, they will
     * be added automatically by the integrator afterwards)
     */
    FieldODEState&lt;T&gt; resetState(FieldODEStateAndDerivative&lt;T&gt; state);
}
</data><data key="type">CompilationUnit</data></node><node id="2"><data key="reference"></data><data key="text">package org.apache.commons.math3.ode.events;

</data><data key="type">PackageDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="3"><data key="reference"></data><data key="identifier">events</data><data key="text">org.apache.commons.math3.ode.events</data><data key="type">Name</data><data key="parentType">PackageDeclaration</data></node><node id="4"><data key="reference"></data><data key="identifier">ode</data><data key="text">org.apache.commons.math3.ode</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="5"><data key="reference"></data><data key="identifier">math3</data><data key="text">org.apache.commons.math3</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="6"><data key="reference"></data><data key="identifier">commons</data><data key="text">org.apache.commons</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="7"><data key="reference"></data><data key="identifier">apache</data><data key="text">org.apache</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="8"><data key="reference"></data><data key="identifier">org</data><data key="text">org</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="9"><data key="reference"></data><data key="text">import org.apache.commons.math3.RealFieldElement;
</data><data key="type">ImportDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="60"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="61"><data key="reference">org.apache.commons.math3.ode.FieldODEStateAndDerivative</data><data key="identifier">state</data><data key="text">state</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="62"><data key="reference"></data><data key="modifier"></data><data key="text">boolean increasing</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="63"><data key="reference"></data><data key="identifier">boolean</data><data key="text">boolean</data><data key="type">PrimitiveType</data><data key="parentType">Parameter</data></node><node id="20"><data key="reference"></data><data key="identifier">apache</data><data key="text">org.apache</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="64"><data key="reference">boolean</data><data key="identifier">increasing</data><data key="text">increasing</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="21"><data key="reference"></data><data key="identifier">org</data><data key="text">org</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="65"><data key="reference"></data><data key="text">Action</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="22"><data key="reference"></data><data key="text">import org.apache.commons.math3.ode.FieldODEStateAndDerivative;
</data><data key="type">ImportDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="66"><data key="reference">nonQualifiedClassName</data><data key="identifier">Action</data><data key="text">Action</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="23"><data key="reference"></data><data key="identifier">FieldODEStateAndDerivative</data><data key="text">org.apache.commons.math3.ode.FieldODEStateAndDerivative</data><data key="type">Name</data><data key="parentType">ImportDeclaration</data></node><node id="67"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * Reset the state prior to continue the integration.
 *
 * &lt;p&gt;This method is called after the step handler has returned and
 * before the next step is started, but only when {@link
 * #eventOccurred(FieldODEStateAndDerivative, boolean) eventOccurred} has itself
 * returned the {@link Action#RESET_STATE} indicator. It allows the user to reset
 * the state vector for the next step, without perturbing the step handler of the
 * finishing step. If the {@link #eventOccurred(FieldODEStateAndDerivative, boolean)
 * eventOccurred} never returns the {@link Action#RESET_STATE} indicator, this
 * function will never be called, and it is safe to leave its body empty.&lt;/p&gt;
 * @param state current value of the independent &lt;i&gt;time&lt;/i&gt; variable, state vector
 * and derivative
 * @return reset state (note that it does not include the derivatives, they will
 * be added automatically by the integrator afterwards)
 */
FieldODEState&lt;T&gt; resetState(FieldODEStateAndDerivative&lt;T&gt; state);</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="24"><data key="reference"></data><data key="identifier">ode</data><data key="text">org.apache.commons.math3.ode</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="68"><data key="reference">userDefinedMethodName</data><data key="identifier">resetState</data><data key="text">resetState</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="25"><data key="reference"></data><data key="identifier">math3</data><data key="text">org.apache.commons.math3</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="69"><data key="reference"></data><data key="modifier"></data><data key="text">FieldODEStateAndDerivative&lt;T&gt; state</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="26"><data key="reference"></data><data key="identifier">commons</data><data key="text">org.apache.commons</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="27"><data key="reference"></data><data key="identifier">apache</data><data key="text">org.apache</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="28"><data key="reference"></data><data key="identifier">org</data><data key="text">org</data><data key="type">Name</data><data key="parentType">Name</data></node><node id="29"><data key="reference"></data><data key="modifier">public,interface</data><data key="text">/**
 * This interface represents a handler for discrete events triggered
 * during ODE integration.
 *
 * &lt;p&gt;Some events can be triggered at discrete times as an ODE problem
 * is solved. This occurs for example when the integration process
 * should be stopped as some state is reached (G-stop facility) when the
 * precise date is unknown a priori, or when the derivatives have
 * discontinuities, or simply when the user wants to monitor some
 * states boundaries crossings.
 * &lt;/p&gt;
 *
 * &lt;p&gt;These events are defined as occurring when a &lt;code&gt;g&lt;/code&gt;
 * switching function sign changes.&lt;/p&gt;
 *
 * &lt;p&gt;Since events are only problem-dependent and are triggered by the
 * independent &lt;i&gt;time&lt;/i&gt; variable and the state vector, they can
 * occur at virtually any time, unknown in advance. The integrators will
 * take care to avoid sign changes inside the steps, they will reduce
 * the step size when such an event is detected in order to put this
 * event exactly at the end of the current step. This guarantees that
 * step interpolation (which always has a one step scope) is relevant
 * even in presence of discontinuities. This is independent from the
 * stepsize control provided by integrators that monitor the local
 * error (this event handling feature is available for all integrators,
 * including fixed step ones).&lt;/p&gt;
 *
 * @param &lt;T&gt; the type of the field elements
 * @since 3.6
 */
public interface FieldEventHandler&lt;T extends RealFieldElement&lt;T&gt;&gt; {

    /**
     * Initialize event handler at the start of an ODE integration.
     * &lt;p&gt;
     * This method is called once at the start of the integration. It
     * may be used by the event handler to initialize some internal data
     * if needed.
     * &lt;/p&gt;
     * @param initialState initial time, state vector and derivative
     * @param finalTime target time for the integration
     */
    void init(FieldODEStateAndDerivative&lt;T&gt; initialState, T finalTime);

    /**
     * Compute the value of the switching function.
     *
     * &lt;p&gt;The discrete events are generated when the sign of this
     * switching function changes. The integrator will take care to change
     * the stepsize in such a way these events occur exactly at step boundaries.
     * The switching function must be continuous in its roots neighborhood
     * (but not necessarily smooth), as the integrator will need to find its
     * roots to locate precisely the events.&lt;/p&gt;
     * &lt;p&gt;Also note that the integrator expect that once an event has occurred,
     * the sign of the switching function at the start of the next step (i.e.
     * just after the event) is the opposite of the sign just before the event.
     * This consistency between the steps &lt;string&gt;must&lt;/strong&gt; be preserved,
     * otherwise {@link org.apache.commons.math3.exception.NoBracketingException
     * exceptions} related to root not being bracketed will occur.&lt;/p&gt;
     * &lt;p&gt;This need for consistency is sometimes tricky to achieve. A typical
     * example is using an event to model a ball bouncing on the floor. The first
     * idea to represent this would be to have {@code g(t) = h(t)} where h is the
     * height above the floor at time {@code t}. When {@code g(t)} reaches 0, the
     * ball is on the floor, so it should bounce and the typical way to do this is
     * to reverse its vertical velocity. However, this would mean that before the
     * event {@code g(t)} was decreasing from positive values to 0, and after the
     * event {@code g(t)} would be increasing from 0 to positive values again.
     * Consistency is broken here! The solution here is to have {@code g(t) = sign
     * * h(t)}, where sign is a variable with initial value set to {@code +1}. Each
     * time {@link #eventOccurred(FieldODEStateAndDerivative, boolean) eventOccurred}
     * method is called, {@code sign} is reset to {@code -sign}. This allows the
     * {@code g(t)} function to remain continuous (and even smooth) even across events,
     * despite {@code h(t)} is not. Basically, the event is used to &lt;em&gt;fold&lt;/em&gt;
     * {@code h(t)} at bounce points, and {@code sign} is used to &lt;em&gt;unfold&lt;/em&gt; it
     * back, so the solvers sees a {@code g(t)} function which behaves smoothly even
     * across events.&lt;/p&gt;
     *
     * @param state current value of the independent &lt;i&gt;time&lt;/i&gt; variable, state vector
     * and derivative
     * @return value of the g switching function
     */
    T g(FieldODEStateAndDerivative&lt;T&gt; state);

    /**
     * Handle an event and choose what to do next.
     *
     * &lt;p&gt;This method is called when the integrator has accepted a step
     * ending exactly on a sign change of the function, just &lt;em&gt;before&lt;/em&gt;
     * the step handler itself is called (see below for scheduling). It
     * allows the user to update his internal data to acknowledge the fact
     * the event has been handled (for example setting a flag in the {@link
     * org.apache.commons.math3.ode.FirstOrderDifferentialEquations
     * differential equations} to switch the derivatives computation in
     * case of discontinuity), or to direct the integrator to either stop
     * or continue integration, possibly with a reset state or derivatives.&lt;/p&gt;
     *
     * &lt;ul&gt;
     *   &lt;li&gt;if {@link Action#STOP} is returned, the step handler will be called
     *   with the &lt;code&gt;isLast&lt;/code&gt; flag of the {@link
     *   org.apache.commons.math3.ode.sampling.StepHandler#handleStep handleStep}
     *   method set to true and the integration will be stopped,&lt;/li&gt;
     *   &lt;li&gt;if {@link Action#RESET_STATE} is returned, the {@link #resetState
     *   resetState} method will be called once the step handler has
     *   finished its task, and the integrator will also recompute the
     *   derivatives,&lt;/li&gt;
     *   &lt;li&gt;if {@link Action#RESET_DERIVATIVES} is returned, the integrator
     *   will recompute the derivatives,
     *   &lt;li&gt;if {@link Action#CONTINUE} is returned, no specific action will
     *   be taken (apart from having called this method) and integration
     *   will continue.&lt;/li&gt;
     * &lt;/ul&gt;
     *
     * &lt;p&gt;The scheduling between this method and the {@link
     * org.apache.commons.math3.ode.sampling.FieldStepHandler FieldStepHandler} method {@link
     * org.apache.commons.math3.ode.sampling.FieldStepHandler#handleStep(
     * org.apache.commons.math3.ode.sampling.FieldStepInterpolator, boolean)
     * handleStep(interpolator, isLast)} is to call this method first and
     * &lt;code&gt;handleStep&lt;/code&gt; afterwards. This scheduling allows the integrator to
     * pass &lt;code&gt;true&lt;/code&gt; as the &lt;code&gt;isLast&lt;/code&gt; parameter to the step
     * handler to make it aware the step will be the last one if this method
     * returns {@link Action#STOP}. As the interpolator may be used to navigate back
     * throughout the last step, user code called by this method and user
     * code called by step handlers may experience apparently out of order values
     * of the independent time variable. As an example, if the same user object
     * implements both this {@link FieldEventHandler FieldEventHandler} interface and the
     * {@link org.apache.commons.math3.ode.sampling.FieldStepHandler FieldStepHandler}
     * interface, a &lt;em&gt;forward&lt;/em&gt; integration may call its
     * {code eventOccurred} method with t = 10 first and call its
     * {code handleStep} method with t = 9 afterwards. Such out of order
     * calls are limited to the size of the integration step for {@link
     * org.apache.commons.math3.ode.sampling.FieldStepHandler variable step handlers}.&lt;/p&gt;
     *
     * @param state current value of the independent &lt;i&gt;time&lt;/i&gt; variable, state vector
     * and derivative
     * @param increasing if true, the value of the switching function increases
     * when times increases around event (note that increase is measured with respect
     * to physical time, not with respect to integration which may go backward in time)
     * @return indication of what the integrator should do next, this
     * value must be one of {@link Action#STOP}, {@link Action#RESET_STATE},
     * {@link Action#RESET_DERIVATIVES} or {@link Action#CONTINUE}
     */
    Action eventOccurred(FieldODEStateAndDerivative&lt;T&gt; state, boolean increasing);

    /**
     * Reset the state prior to continue the integration.
     *
     * &lt;p&gt;This method is called after the step handler has returned and
     * before the next step is started, but only when {@link
     * #eventOccurred(FieldODEStateAndDerivative, boolean) eventOccurred} has itself
     * returned the {@link Action#RESET_STATE} indicator. It allows the user to reset
     * the state vector for the next step, without perturbing the step handler of the
     * finishing step. If the {@link #eventOccurred(FieldODEStateAndDerivative, boolean)
     * eventOccurred} never returns the {@link Action#RESET_STATE} indicator, this
     * function will never be called, and it is safe to leave its body empty.&lt;/p&gt;
     * @param state current value of the independent &lt;i&gt;time&lt;/i&gt; variable, state vector
     * and derivative
     * @return reset state (note that it does not include the derivatives, they will
     * be added automatically by the integrator afterwards)
     */
    FieldODEState&lt;T&gt; resetState(FieldODEStateAndDerivative&lt;T&gt; state);
}</data><data key="type">ClassOrInterfaceDeclaration</data><data key="parentType">CompilationUnit</data></node><node id="70"><data key="reference"></data><data key="text">FieldODEStateAndDerivative&lt;T&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="71"><data key="reference">nonQualifiedClassName</data><data key="identifier">FieldODEStateAndDerivative</data><data key="text">FieldODEStateAndDerivative</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="72"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="73"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="30"><data key="reference"></data><data key="identifier">FieldEventHandler</data><data key="text">FieldEventHandler</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="74"><data key="reference">org.apache.commons.math3.ode.FieldODEStateAndDerivative</data><data key="identifier">state</data><data key="text">state</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="31"><data key="reference"></data><data key="modifier"></data><data key="text">/**
 * Initialize event handler at the start of an ODE integration.
 * &lt;p&gt;
 * This method is called once at the start of the integration. It
 * may be used by the event handler to initialize some internal data
 * if needed.
 * &lt;/p&gt;
 * @param initialState initial time, state vector and derivative
 * @param finalTime target time for the integration
 */
void init(FieldODEStateAndDerivative&lt;T&gt; initialState, T finalTime);</data><data key="type">MethodDeclaration</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="75"><data key="reference"></data><data key="text">FieldODEState&lt;T&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">MethodDeclaration</data></node><node id="32"><data key="reference">userDefinedMethodName</data><data key="identifier">init</data><data key="text">init</data><data key="type">SimpleName</data><data key="parentType">MethodDeclaration</data></node><node id="76"><data key="reference">nonQualifiedClassName</data><data key="identifier">FieldODEState</data><data key="text">FieldODEState</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="33"><data key="reference"></data><data key="modifier"></data><data key="text">FieldODEStateAndDerivative&lt;T&gt; initialState</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="77"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="34"><data key="reference"></data><data key="text">FieldODEStateAndDerivative&lt;T&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="78"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="35"><data key="reference">nonQualifiedClassName</data><data key="identifier">FieldODEStateAndDerivative</data><data key="text">FieldODEStateAndDerivative</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="79"><data key="reference"></data><data key="text">T extends RealFieldElement&lt;T&gt;</data><data key="type">TypeParameter</data><data key="parentType">ClassOrInterfaceDeclaration</data></node><node id="36"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="37"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="38"><data key="reference">org.apache.commons.math3.ode.FieldODEStateAndDerivative</data><data key="identifier">initialState</data><data key="text">initialState</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="39"><data key="reference"></data><data key="modifier"></data><data key="text">T finalTime</data><data key="type">Parameter</data><data key="parentType">MethodDeclaration</data></node><node id="80"><data key="reference">runtimeGenericType</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">TypeParameter</data></node><node id="81"><data key="reference"></data><data key="text">RealFieldElement&lt;T&gt;</data><data key="type">ClassOrInterfaceType</data><data key="parentType">TypeParameter</data></node><node id="82"><data key="reference">nonQualifiedClassName</data><data key="identifier">RealFieldElement</data><data key="text">RealFieldElement</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="83"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">ClassOrInterfaceType</data></node><node id="40"><data key="reference"></data><data key="text">T</data><data key="type">ClassOrInterfaceType</data><data key="parentType">Parameter</data></node><node id="84"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="41"><data key="reference">nonQualifiedClassName</data><data key="identifier">T</data><data key="text">T</data><data key="type">SimpleName</data><data key="parentType">ClassOrInterfaceType</data></node><node id="42"><data key="reference">genericType</data><data key="identifier">finalTime</data><data key="text">finalTime</data><data key="type">SimpleName</data><data key="parentType">Parameter</data></node><node id="43"><data key="reference"></data><data key="text">void</data><data key="type">VoidType</data><data key="parentType">MethodDeclaration</data></node><edge id="56" source="44" target="45" label="AST"><data key="type">AST</data></edge><edge id="67" source="44" target="52" label="AST"><data key="type">AST</data></edge><edge id="58" source="44" target="46" label="AST"><data key="type">AST</data></edge><edge id="55" source="44" target="54" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="57" source="45" target="46" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="66" source="46" target="51" label="AST"><data key="type">AST</data></edge><edge id="60" source="46" target="47" label="AST"><data key="type">AST</data></edge><edge id="59" source="46" target="52" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="62" source="47" target="48" label="AST"><data key="type">AST</data></edge><edge id="64" source="47" target="49" label="AST"><data key="type">AST</data></edge><edge id="61" source="47" target="51" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="63" source="48" target="49" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="65" source="49" target="50" label="AST"><data key="type">AST</data></edge><edge id="68" source="52" target="53" label="AST"><data key="type">AST</data></edge><edge id="11" source="10" target="11" label="AST"><data key="type">AST</data></edge><edge id="71" source="54" target="55" label="AST"><data key="type">AST</data></edge><edge id="82" source="54" target="62" label="AST"><data key="type">AST</data></edge><edge id="73" source="54" target="56" label="AST"><data key="type">AST</data></edge><edge id="87" source="54" target="65" label="AST"><data key="type">AST</data></edge><edge id="70" source="54" target="67" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="12" source="11" target="12" label="AST"><data key="type">AST</data></edge><edge id="72" source="55" target="56" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="13" source="12" target="13" label="AST"><data key="type">AST</data></edge><edge id="81" source="56" target="61" label="AST"><data key="type">AST</data></edge><edge id="75" source="56" target="57" label="AST"><data key="type">AST</data></edge><edge id="74" source="56" target="62" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="14" source="13" target="14" label="AST"><data key="type">AST</data></edge><edge id="77" source="57" target="58" label="AST"><data key="type">AST</data></edge><edge id="79" source="57" target="59" label="AST"><data key="type">AST</data></edge><edge id="76" source="57" target="61" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="78" source="58" target="59" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="17" source="15" target="16" label="AST"><data key="type">AST</data></edge><edge id="16" source="15" target="22" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="80" source="59" target="60" label="AST"><data key="type">AST</data></edge><edge id="18" source="16" target="17" label="AST"><data key="type">AST</data></edge><edge id="19" source="17" target="18" label="AST"><data key="type">AST</data></edge><edge id="20" source="18" target="19" label="AST"><data key="type">AST</data></edge><edge id="21" source="19" target="20" label="AST"><data key="type">AST</data></edge><edge id="0" source="1" target="2" label="AST"><data key="type">AST</data></edge><edge id="23" source="1" target="22" label="AST"><data key="type">AST</data></edge><edge id="15" source="1" target="15" label="AST"><data key="type">AST</data></edge><edge id="8" source="1" target="9" label="AST"><data key="type">AST</data></edge><edge id="31" source="1" target="29" label="AST"><data key="type">AST</data></edge><edge id="2" source="2" target="3" label="AST"><data key="type">AST</data></edge><edge id="1" source="2" target="9" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="3" source="3" target="4" label="AST"><data key="type">AST</data></edge><edge id="4" source="4" target="5" label="AST"><data key="type">AST</data></edge><edge id="5" source="5" target="6" label="AST"><data key="type">AST</data></edge><edge id="6" source="6" target="7" label="AST"><data key="type">AST</data></edge><edge id="7" source="7" target="8" label="AST"><data key="type">AST</data></edge><edge id="10" source="9" target="10" label="AST"><data key="type">AST</data></edge><edge id="9" source="9" target="15" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="84" source="62" target="63" label="AST"><data key="type">AST</data></edge><edge id="86" source="62" target="64" label="AST"><data key="type">AST</data></edge><edge id="83" source="62" target="65" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="85" source="63" target="64" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="22" source="20" target="21" label="AST"><data key="type">AST</data></edge><edge id="88" source="65" target="66" label="AST"><data key="type">AST</data></edge><edge id="25" source="22" target="23" label="AST"><data key="type">AST</data></edge><edge id="24" source="22" target="29" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="26" source="23" target="24" label="AST"><data key="type">AST</data></edge><edge id="102" source="67" target="75" label="AST"><data key="type">AST</data></edge><edge id="91" source="67" target="68" label="AST"><data key="type">AST</data></edge><edge id="93" source="67" target="69" label="AST"><data key="type">AST</data></edge><edge id="90" source="67" target="79" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="27" source="24" target="25" label="AST"><data key="type">AST</data></edge><edge id="92" source="68" target="69" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="28" source="25" target="26" label="AST"><data key="type">AST</data></edge><edge id="101" source="69" target="74" label="AST"><data key="type">AST</data></edge><edge id="95" source="69" target="70" label="AST"><data key="type">AST</data></edge><edge id="94" source="69" target="75" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="29" source="26" target="27" label="AST"><data key="type">AST</data></edge><edge id="30" source="27" target="28" label="AST"><data key="type">AST</data></edge><edge id="34" source="29" target="31" label="AST"><data key="type">AST</data></edge><edge id="89" source="29" target="67" label="AST"><data key="type">AST</data></edge><edge id="69" source="29" target="54" label="AST"><data key="type">AST</data></edge><edge id="107" source="29" target="79" label="AST"><data key="type">AST</data></edge><edge id="32" source="29" target="30" label="AST"><data key="type">AST</data></edge><edge id="54" source="29" target="44" label="AST"><data key="type">AST</data></edge><edge id="99" source="70" target="72" label="AST"><data key="type">AST</data></edge><edge id="97" source="70" target="71" label="AST"><data key="type">AST</data></edge><edge id="96" source="70" target="74" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="98" source="71" target="72" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="100" source="72" target="73" label="AST"><data key="type">AST</data></edge><edge id="33" source="30" target="31" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="36" source="31" target="32" label="AST"><data key="type">AST</data></edge><edge id="47" source="31" target="39" label="AST"><data key="type">AST</data></edge><edge id="38" source="31" target="33" label="AST"><data key="type">AST</data></edge><edge id="53" source="31" target="43" label="AST"><data key="type">AST</data></edge><edge id="35" source="31" target="44" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="103" source="75" target="76" label="AST"><data key="type">AST</data></edge><edge id="105" source="75" target="77" label="AST"><data key="type">AST</data></edge><edge id="37" source="32" target="33" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="104" source="76" target="77" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="46" source="33" target="38" label="AST"><data key="type">AST</data></edge><edge id="40" source="33" target="34" label="AST"><data key="type">AST</data></edge><edge id="39" source="33" target="39" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="106" source="77" target="78" label="AST"><data key="type">AST</data></edge><edge id="44" source="34" target="36" label="AST"><data key="type">AST</data></edge><edge id="42" source="34" target="35" label="AST"><data key="type">AST</data></edge><edge id="41" source="34" target="38" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="43" source="35" target="36" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="110" source="79" target="81" label="AST"><data key="type">AST</data></edge><edge id="108" source="79" target="80" label="AST"><data key="type">AST</data></edge><edge id="45" source="36" target="37" label="AST"><data key="type">AST</data></edge><edge id="49" source="39" target="40" label="AST"><data key="type">AST</data></edge><edge id="52" source="39" target="42" label="AST"><data key="type">AST</data></edge><edge id="48" source="39" target="43" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="109" source="80" target="81" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="111" source="81" target="82" label="AST"><data key="type">AST</data></edge><edge id="113" source="81" target="83" label="AST"><data key="type">AST</data></edge><edge id="112" source="82" target="83" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge><edge id="114" source="83" target="84" label="AST"><data key="type">AST</data></edge><edge id="51" source="40" target="41" label="AST"><data key="type">AST</data></edge><edge id="50" source="40" target="42" label="NEXT_TOKEN"><data key="type">NEXT_TOKEN</data></edge></graph></graphml>