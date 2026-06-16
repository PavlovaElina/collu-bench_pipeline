#lang racket
(define (circular_shift x shift)
  (define s (number->string x))
  (define n (string-length s))
  (cond [(> shift n) (list->string (reverse (string->list s)))]
        [else (define k (modulo shift n))
              (string-append (substring s (- n k) n) (substring s 0 (- n k)))]))
