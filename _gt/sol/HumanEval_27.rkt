#lang racket
(define (flip_case s)
  (list->string
   (map (lambda (c)
          (cond [(char-upper-case? c) (char-downcase c)]
                [(char-lower-case? c) (char-upcase c)]
                [else c]))
        (string->list s))))
