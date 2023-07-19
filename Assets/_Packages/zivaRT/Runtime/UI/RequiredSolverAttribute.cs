using UnityEngine;

internal class RequiredImplementationAttribute : PropertyAttribute
{
    public ZivaRTPlayer.ImplementationType implementation;

    public RequiredImplementationAttribute(ZivaRTPlayer.ImplementationType implementation) { this.implementation = implementation; }
}
