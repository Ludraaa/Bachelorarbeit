import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpAsQuery;
import org.apache.jena.sparql.sse.SSE;

public class JenaServer {

    static String decode(String s) {
        return new String(
            Base64.getDecoder().decode(s),
            StandardCharsets.UTF_8
        );
    }

    static String encode(String s) {
        return Base64.getEncoder().encodeToString(
            s.getBytes(StandardCharsets.UTF_8)
        );
    }

    public static void main(String[] args) throws Exception {

        BufferedReader reader =
            new BufferedReader(new InputStreamReader(System.in));

        BufferedWriter writer =
            new BufferedWriter(new OutputStreamWriter(System.out));

        String cmd;

        while ((cmd = reader.readLine()) != null) {

            String encoded = reader.readLine();

            if (encoded == null)
                break;

            String data = decode(encoded);

            try {

                String result;

                switch (cmd) {

                    case "sparql_to_algebra": {

                        Query query = QueryFactory.create(data);

                        Op op = Algebra.compile(query);

                        result = op.toString();

                        break;
                    }

                    case "algebra_to_sparql": {

                        Op op = SSE.parseOp(data);

                        Query query = OpAsQuery.asQuery(op);

                        result = query.serialize();

                        break;
                    }

                    default:
                        throw new RuntimeException(
                            "Unknown command: " + cmd
                        );
                }

                writer.write("OK\n");
                writer.write(encode(result) + "\n");
                writer.flush();

            } catch (Exception e) {

                StringWriter sw = new StringWriter();
                e.printStackTrace(new PrintWriter(sw));

                writer.write("ERR\n");
                writer.write(encode(sw.toString()) + "\n");
                writer.flush();
            }
        }
    }
}