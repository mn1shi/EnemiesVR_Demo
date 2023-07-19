using UnityEditor;
using UnityEngine;

[CustomPropertyDrawer(typeof(RequiredImplementationAttribute))]
internal class RequiredSolverAttributeDrawer : PropertyDrawer
{
    bool m_ShowProperty = true;
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        RequiredImplementationAttribute attr = attribute as RequiredImplementationAttribute;
        // If the name of this variable changes, this code will break.
        SerializedProperty solverProp = property.serializedObject.FindProperty("m_Implementation");
        ZivaRTPlayer.ImplementationType componentsSolverType = (ZivaRTPlayer.ImplementationType)solverProp.intValue;
        m_ShowProperty = (componentsSolverType == attr.implementation);
        if (m_ShowProperty)
        {
            EditorGUI.PropertyField(position, property, label);
        }
    }

    // This prevents an empty space remaining if the property is hidden.
    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        if (m_ShowProperty)
        {
            return base.GetPropertyHeight(property, label);
        }
        else
        {
            return 0.0f;
        }
    }
}
