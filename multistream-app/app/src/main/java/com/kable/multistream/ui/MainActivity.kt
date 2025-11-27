package com.kable.multistream.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.kable.multistream.ui.theme.MultiStreamTheme
import timber.log.Timber

class MainActivity : ComponentActivity() {

    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    )

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            Timber.d("All permissions granted")
        } else {
            Timber.w("Some permissions were denied")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Check and request permissions
        checkPermissions()

        setContent {
            MultiStreamTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }
    }

    private fun checkPermissions() {
        val permissionsToRequest = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (permissionsToRequest.isNotEmpty()) {
            permissionLauncher.launch(permissionsToRequest.toTypedArray())
        }
    }
}

@Composable
fun MainScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(PaddingValues())
    ) {
        Text(
            text = "MultiStream",
            style = MaterialTheme.typography.headlineLarge,
            modifier = Modifier.padding(all = androidx.compose.ui.unit.dp(16f))
        )
        Text(
            text = "Multi-platform live streaming app",
            style = MaterialTheme.typography.bodyLarge,
            modifier = Modifier.padding(horizontal = androidx.compose.ui.unit.dp(16f))
        )
        Spacer(modifier = Modifier.height(androidx.compose.ui.unit.dp(24f)))
        Text(
            text = "Coming soon: Connect your platforms and go live!",
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(horizontal = androidx.compose.ui.unit.dp(16f))
        )
    }
}
