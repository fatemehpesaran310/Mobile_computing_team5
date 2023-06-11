package com.example.proj_interface;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.TextView;
import android.view.View;

import com.google.android.material.textfield.TextInputEditText;
import com.google.android.material.textfield.TextInputLayout;

import org.w3c.dom.Text;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        String curr_caption = "Write summarized caption";

        TextView caption_view = (TextView) findViewById(R.id.caption);
        caption_view.setText(curr_caption);




    }

    public void sendbutton_click(View view){
        TextInputLayout text_input = (TextInputLayout) findViewById(R.id.text_input);
        String edited_text = text_input.getEditText().getText().toString().trim();

        TextView caption_view = (TextView) findViewById(R.id.caption);
        caption_view.setText(edited_text);

    }


}